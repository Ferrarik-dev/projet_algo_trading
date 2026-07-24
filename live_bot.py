import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from transformers import pipeline
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import warnings
import json
import requests

load_dotenv()
try:
    client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
except Exception as e:
    print(f"Erreur d'initialisation Alpaca: {e}")
    client = None

warnings.filterwarnings('ignore')

# --- CONFIGURATION (STEFAN JANSEN SETUP) ---
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'BAC', 'COST'
]
TOP_N = 3 # On garde le Top 3 qui est le plus performant
HORIZON_DAYS = 5

def send_telegram_message(message):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("Identifiants Telegram manquants dans l'environnement.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erreur d'envoi Telegram : {e}")

def prepare_features(df_raw):
    df = df_raw.copy()
    
    # Lags (Momentum)
    for lag in [1, 5, 21, 42, 63, 126, 252]:
        df[f'Ret_{lag}d'] = df['Close'].pct_change(lag)
        
    df['Vol_21d'] = df['Ret_1d'].rolling(21).std()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['Dist_BB_Upper'] = (df['Close'] / df['BB_Upper']) - 1
    df['Dist_BB_Lower'] = (df['Close'] / df['BB_Lower']) - 1
    
    # Time Dummies
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    # Target
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df.dropna(inplace=True)
    return df

def generate_todays_signals():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TELECHARGEMENT DES DONNEES (TOP 20 S&P500)")
    data_dict = {}
    for ticker in UNIVERSE:
        df = yf.download(ticker, period='5y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            data_dict[ticker] = df

    print("PREPARATION DES FEATURES...")
    all_features = {}
    features_list = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
    
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df)

    print("ENTRAINEMENT DU MODELE LIGHTGBM (sans look-ahead bias)...")
    X_train_list, y_train_list = [], []
    current_date = list(all_features.values())[0].index[-1]
    
    for ticker, df in all_features.items():
        # Retrait des 5 derniers jours pour eviter la triche sur le Target_5d
        df_hist = df.iloc[:-5] 
        X_train_list.append(df_hist[features_list])
        y_train_list.append(df_hist['Target_5d'])
        
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        num_leaves=31, subsample=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("GENERATION DES PREDICTIONS POUR AUJOURD'HUI...")
    predictions = []
    for ticker, df in all_features.items():
        row = df.iloc[-1]
        if not row[features_list].isnull().any():
            X_today = pd.DataFrame([row[features_list]])
            pred_return = model.predict(X_today)[0]
            if pred_return > 0: # On ne s'interesse qu'aux actions prevues a la hausse
                predictions.append((ticker, pred_return))
                
    # Classement
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_picks = predictions[:TOP_N]
    
    # Construction de la Market Data pour le Dashboard
    dashboard_market = {}
    for ticker, df in data_dict.items():
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        pct_change = ((last_close / prev_close) - 1) * 100
        dashboard_market[ticker] = {
            'price': float(last_close),
            'change_pct': float(pct_change)
        }
    
    print("\n--- ANALYSE NLP FINBERT (Mode Alerte) ---")
    print("Chargement du modele linguistique...")
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    top_picks_dashboard = []
    
    if not top_picks:
        print("Aucun signal d'achat interessant aujourd'hui.")
    else:
        print("\n--- MESSAGE TELEGRAM DE SYNTHESE ---")
        telegram_msg = "🤖 [QuantBot Premium] - Rapport du Jour\n\n"
        for rank, (ticker, pred) in enumerate(top_picks, 1):
            
            # NLP Scoring
            stock = yf.Ticker(ticker)
            news_items = stock.news
            headlines = []
            if news_items:
                for item in news_items[:5]:
                    if 'content' in item and 'title' in item['content']:
                        headlines.append(item['content']['title'])
                    elif 'title' in item:
                        headlines.append(item['title'])
            
            avg_score = 0
            if headlines:
                total_score = 0
                for headline in headlines:
                    res = nlp(headline)[0]
                    score = res['score'] * (1 if res['label'] == 'positive' else -1 if res['label'] == 'negative' else 0)
                    total_score += score
                avg_score = total_score / len(headlines)
            
            # Shadow Mode Logic
            nlp_alert = ""
            if avg_score < -0.2:
                nlp_alert = f"ATTENTION: Score Media Catastrophique ({avg_score:.2f}). J'AURAIS MIS MON VETO !"
            elif avg_score > 0.2:
                nlp_alert = f"FEU VERT: Score Media Positif ({avg_score:.2f})."
            else:
                nlp_alert = f"NEUTRE: Score Media ({avg_score:.2f})."
                
            line_str = f"#{rank} {ticker} (Rendement 5J prevu : +{pred*100:.2f}%)\n   -> NLP Opinion: {nlp_alert}"
            print(line_str)
            telegram_msg += line_str + "\n\n"
            
            top_picks_dashboard.append({
                'ticker': ticker,
                'pred_return': float(pred * 100),
                'nlp_score': float(avg_score),
                'nlp_alert': nlp_alert,
                'headlines': headlines
            })
            
        send_telegram_message(telegram_msg)
            
    print("-" * 45)
    return top_picks, dashboard_market, top_picks_dashboard

def format_ticker_for_alpaca(ticker):
    # Les actions standards n'ont pas besoin de formatage particulier
    return ticker

def execute_live_orders(buy_signals):
    account_info = {'balance': 0.0, 'buying_power': 0.0}
    if client is None:
        print("API Alpaca non configuree. Mode Simulation uniquement.")
        return account_info
        
    print("\nEtape 3 : Execution des Ordres sur Alpaca...")
    
    # 0. Annuler tous les ordres en attente
    try:
        client.cancel_orders()
        print("Nettoyage : Tous les anciens ordres en attente ont ete annules.")
    except Exception as e:
        print(f"Erreur lors de l'annulation des ordres : {e}")
    
    # 1. Lister les positions actuelles
    positions = client.get_all_positions()
    current_holdings = {pos.symbol: float(pos.qty) for pos in positions}
    
    print(f"Positions actuellement detenues : {list(current_holdings.keys())}")
    print(f"Nouveaux Signaux d'Achat generes par l'IA : {buy_signals}")
    
    alpaca_buy_signals = [format_ticker_for_alpaca(t) for t in buy_signals]
    
    # 2. Vendre ce qui n'a plus de signal
    for symbol in current_holdings.keys():
        if symbol not in alpaca_buy_signals:
            print(f"Vente de {symbol} (Sorti du Top-3)")
            try:
                client.close_position(symbol)
            except Exception as e:
                print(f"Erreur a la revente de {symbol} : {e}")
            
    # 3. Acheter les nouveaux signaux
    if len(alpaca_buy_signals) == 0:
        print("Aucun signal d'achat aujourd'hui. L'IA reste en securite (Cash).")
        return
        
    # Smart Rebalance : Identifier les NOUVEAUX actifs a acheter
    new_assets = [s for s in alpaca_buy_signals if s not in current_holdings]
    
    if len(new_assets) > 0:
        account = client.get_account()
        buying_power = float(account.buying_power)
        # On utilise 95% du cash disponible, divise par le nombre de NOUVEAUX actifs a financer
        budget_per_asset = (buying_power * 0.95) / len(new_assets)
        budget_per_asset = round(budget_per_asset, 2)
        print(f"Budget alloue par NOUVEL actif : {budget_per_asset} $")
    else:
        print("Tous les actifs du Top-3 sont deja en portefeuille. Aucun nouvel achat necessaire.")
        budget_per_asset = 0
    
    for symbol in new_assets:
        print(f"Achat de {symbol}")
        try:
            # Ordre Notionnel (Base sur le budget en $, fractionnel automatique)
            req = MarketOrderRequest(
                symbol=symbol,
                notional=budget_per_asset,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(req)
            print(f" -> Ordre execute : {budget_per_asset:.2f} $ de {symbol}")
        except Exception as e:
            print(f"Erreur lors de l'achat de {symbol} : {e}")

    # Return account info for Dashboard
    try:
        acc = client.get_account()
        account_info['balance'] = float(acc.portfolio_value)
        account_info['buying_power'] = float(acc.buying_power)
    except:
        pass
    return account_info

if __name__ == '__main__':
    top_picks, dashboard_market, top_picks_dashboard = generate_todays_signals()
    buy_signals = [item[0] for item in top_picks] if top_picks else []
    
    account_info = execute_live_orders(buy_signals)
    
    # Generation du JSON pour le Dashboard
    dashboard_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'account': account_info,
        'market_data': dashboard_market,
        'top_picks': top_picks_dashboard
    }
    
    with open('dashboard_data.json', 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=4, ensure_ascii=False)
        
    print("\n[OK] Donnees du Dashboard exportees (dashboard_data.json)")
    print("Termine pour aujourd'hui ! Le robot a ferme ses portes.")
