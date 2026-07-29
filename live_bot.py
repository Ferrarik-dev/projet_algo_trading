import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from transformers import pipeline
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
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

FEATURES_LIST = [
    'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
    'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month',
    'Volume_Ratio', 'ATR_14', 'OBV', 'SMA200_dist', 'VIX', 'Sector_RS'
]

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

def prepare_features(df_raw, vix_data=None, spy_data=None, for_prediction=False):
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
    
    # Nouveaux indicateurs
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR_14'] = atr / df['Close']
    
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
    
    sma200 = df['Close'].rolling(200).mean()
    df['SMA200_dist'] = (df['Close'] / sma200) - 1
    
    if vix_data is not None:
        df['VIX'] = vix_data['Close'].reindex(df.index).ffill()
    else:
        df['VIX'] = 20.0
        
    if spy_data is not None:
        spy_ret21 = spy_data['Close'].pct_change(21)
        spy_ret21_aligned = spy_ret21.reindex(df.index).ffill()
        df['Sector_RS'] = df['Ret_21d'] - spy_ret21_aligned
    else:
        df['Sector_RS'] = 0.0
    
    if not for_prediction:
        # Target
        df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df.dropna(inplace=True)
    else:
        df.dropna(subset=FEATURES_LIST, inplace=True)
        
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

    # Telechargement VIX et SPY
    vix_df = yf.download('^VIX', period='5y', progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)
    
    spy_df = yf.download('SPY', period='5y', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)

    print("PREPARATION DES FEATURES...")
    all_features_train = {}
    all_features_predict = {}
    
    for ticker, df in data_dict.items():
        all_features_train[ticker] = prepare_features(df, vix_df, spy_df, for_prediction=False)
        all_features_predict[ticker] = prepare_features(df, vix_df, spy_df, for_prediction=True)

    print("ENTRAINEMENT DU MODELE LIGHTGBM ET META-MODELE (sans look-ahead bias)...")
    X_train_list, y_train_list, meta_features_list = [], [], []
    
    for ticker, df in all_features_train.items():
        # Retrait des 5 derniers jours pour eviter la triche sur le Target_5d
        df_hist = df.iloc[:-5] 
        X_train_list.append(df_hist[FEATURES_LIST])
        y_train_list.append(df_hist['Target_5d'])
        meta_features_list.append(df_hist[['Vol_21d', 'VIX', 'Sector_RS']])
        
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    X_train_sorted = X_train.sort_index()
    y_train_sorted = y_train.loc[X_train_sorted.index]
    meta_features_sorted = pd.concat(meta_features_list).loc[X_train_sorted.index]
    
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        num_leaves=31, subsample=0.8, random_state=42, n_jobs=-1
    )
    
    from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    
    cv = TimeSeriesSplit(n_splits=5)
    primary_preds = cross_val_predict(model, X_train_sorted, y_train_sorted, cv=cv, n_jobs=-1)
    
    X_meta_train = meta_features_sorted.copy()
    X_meta_train['Primary_Pred'] = primary_preds
    y_meta_train = (y_train_sorted > 0).astype(int)
    
    meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    meta_model.fit(X_meta_train, y_meta_train)
    
    model.fit(X_train, y_train)
    
    print("GENERATION DES PREDICTIONS POUR AUJOURD'HUI...")
    predictions = []
    full_universe_analysis = []
    
    for ticker, df_predict in all_features_predict.items():
        row = df_predict.iloc[-1]
        
        df_raw = data_dict[ticker]
        last_close = df_raw['Close'].iloc[-1]
        prev_close = df_raw['Close'].iloc[-2]
        change_pct = ((last_close / prev_close) - 1) * 100
        
        pred_return = 0
        meta_prob = 0.0
        if not row[FEATURES_LIST].isnull().any():
            X_today = pd.DataFrame([row[FEATURES_LIST]])
            pred_return = float(model.predict(X_today)[0])
            
            if pred_return > 0: # On ne s'interesse qu'aux actions prevues a la hausse pour le Top 3
                X_meta_today = pd.DataFrame([{
                    'Vol_21d': row['Vol_21d'],
                    'VIX': row['VIX'],
                    'Sector_RS': row['Sector_RS'],
                    'Primary_Pred': pred_return
                }])
                meta_prob = float(meta_model.predict_proba(X_meta_today)[0][1])
                
                if meta_prob >= 0.55:
                    predictions.append((ticker, pred_return, meta_prob))
                
        full_universe_analysis.append({
            'ticker': ticker,
            'price': float(last_close),
            'change_pct': float(change_pct),
            'rsi': float(row['RSI_14']) if not pd.isna(row['RSI_14']) else 0.0,
            'macd': float(row['MACD']) if not pd.isna(row['MACD']) else 0.0,
            'volatility': float(row['Vol_21d'] * 100) if not pd.isna(row['Vol_21d']) else 0.0,
            'pred_return': float(pred_return * 100),
            'meta_prob': float(meta_prob * 100)
        })
                
    # Classement
    predictions.sort(key=lambda x: x[1], reverse=True)
    full_universe_analysis.sort(key=lambda x: x['pred_return'], reverse=True)
    top_picks = predictions[:TOP_N]
    
    print("\n--- ANALYSE NLP FINBERT (Mode Alerte) ---")
    print("Chargement du modele linguistique...")
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    top_picks_details = []
    if not top_picks:
        print("Aucun signal d'achat interessant aujourd'hui.")
    else:
        for rank, (ticker, pred, meta_prob) in enumerate(top_picks, 1):
            
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
                nlp_alert = f"ATTENTION (Catastrophique)"
            elif avg_score > 0.2:
                nlp_alert = f"FEU VERT (Positif)"
            else:
                nlp_alert = f"NEUTRE"
                
            top_picks_details.append({
                'ticker': ticker,
                'pred_return': pred,
                'meta_prob': meta_prob,
                'nlp_score': avg_score,
                'nlp_alert': nlp_alert
            })
            
    print("-" * 45)
    return top_picks_details, full_universe_analysis



def get_performance_comparison(alpaca_client):
    perf_data = {
        'timestamps': [],
        'bot_equity': [],
        'spy_equity': [],
        'bot_return_pct': 0.0,
        'spy_return_pct': 0.0
    }
    
    if alpaca_client is None:
        return perf_data
        
    print("\nEtape 4 : Calcul des performances (Bot vs S&P 500)...")
    try:
        req = GetPortfolioHistoryRequest(period="all", timeframe="1D")
        history = alpaca_client.get_portfolio_history(req)
        
        if not history.timestamp or len(history.timestamp) == 0:
            return perf_data
            
        timestamps = history.timestamp
        equity = history.equity
        
        date_strs = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
        start_date = date_strs[0]
        end_date_obj = datetime.strptime(date_strs[-1], '%Y-%m-%d') + pd.Timedelta(days=1)
        end_date = end_date_obj.strftime('%Y-%m-%d')
        
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
            
        if spy.empty:
            return perf_data
            
        initial_bot_equity = equity[0]
        initial_spy_price = spy['Close'].iloc[0]
        
        bot_normalized = [(e / initial_bot_equity) * 100 for e in equity]
        spy_normalized = []
        
        for d in date_strs:
            if d in spy.index.strftime('%Y-%m-%d'):
                price = spy.loc[d, 'Close']
                spy_normalized.append((price / initial_spy_price) * 100)
            else:
                spy_normalized.append(spy_normalized[-1] if len(spy_normalized) > 0 else 100)
                
        perf_data['timestamps'] = date_strs
        perf_data['bot_equity'] = bot_normalized
        perf_data['spy_equity'] = spy_normalized
        perf_data['bot_return_pct'] = ((equity[-1] / initial_bot_equity) - 1) * 100
        perf_data['spy_return_pct'] = (spy_normalized[-1] - 100)
        
    except Exception as e:
        print(f"Erreur calcul de performance : {e}")
        
    return perf_data

def format_ticker_for_alpaca(ticker):
    # Les actions standards n'ont pas besoin de formatage particulier
    return ticker

def execute_live_orders(top_picks_details, trade_today=True):
    if client is None:
        print("API Alpaca non configuree. Mode Simulation uniquement.")
        return
        
    print("\nEtape 3 : Execution des Ordres sur Alpaca...")
    
    # 0. Etat Initial du compte
    account = client.get_account()
    initial_balance = float(account.portfolio_value)
    
    try:
        client.cancel_orders()
        print("Nettoyage : Tous les anciens ordres en attente ont ete annules.")
    except Exception as e:
        print(f"Erreur lors de l'annulation des ordres : {e}")
    
    # 1. Lister les positions actuelles (Avant changement)
    positions = client.get_all_positions()
    current_holdings = {pos.symbol: float(pos.qty) for pos in positions}
    holdings_before = list(current_holdings.keys())
    
    buy_signals = [pick['ticker'] for pick in top_picks_details]
    alpaca_buy_signals = [format_ticker_for_alpaca(t) for t in buy_signals]
    
    print(f"Positions actuellement detenues : {holdings_before}")
    if trade_today:
        print(f"Nouveaux Signaux d'Achat generes par l'IA : {buy_signals}")
    else:
        print(f"Signaux d'Achat (Information uniquement) : {buy_signals}")
        
    if not trade_today:
        print("--- REBALANCEMENT SUSPENDU ---")
        print("La derniere session n'etait pas un Lundi. Mise a jour du Dashboard uniquement.")
    else:
        # 2. Vendre ce qui n'a plus de signal
        for symbol in current_holdings.keys():
            if symbol not in alpaca_buy_signals:
                print(f"Vente de {symbol} (Sorti du Top-3)")
                try:
                    client.close_position(symbol)
                except Exception as e:
                    print(f"Erreur a la revente de {symbol} : {e}")
            
    # 3. Acheter les nouveaux signaux
    if trade_today:
        if len(alpaca_buy_signals) > 0:
            new_assets = [s for s in alpaca_buy_signals if s not in current_holdings]
            
            if len(new_assets) > 0:
                # On utilise le account reactualise
                account_fresh = client.get_account()
                equity = float(account_fresh.portfolio_value)
                
                # Objectif de levier strict : x1.95 maximum pour eviter l'Appel de Marge de nuit
                target_leverage = 1.95
                budget_per_asset = (equity * target_leverage) / TOP_N
                budget_per_asset = round(budget_per_asset, 2)
                
                print(f"Budget alloue par NOUVEL actif : {budget_per_asset} $ (Basé sur un Levier x{target_leverage})")
                
                for symbol in new_assets:
                    print(f"Achat de {symbol}")
                    try:
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
            else:
                print("Tous les actifs du Top-3 sont deja en portefeuille. Aucun nouvel achat necessaire.")
        else:
            print("Aucun signal d'achat aujourd'hui. L'IA reste en securite (Cash).")

    # 4. Bilan Post-Execution pour Telegram
    print("\n--- PREPARATION DU MESSAGE TELEGRAM ---")
    new_positions = client.get_all_positions()
    new_holdings = [pos.symbol for pos in new_positions]
    
    investments = {}
    prices = {}
    for pos in new_positions:
        investments[pos.symbol] = float(pos.market_value)
        prices[pos.symbol] = float(pos.current_price)
        
    telegram_msg = "🤖 *[QuantBot Premium]* - Rapport du Jour\n\n"
    telegram_msg += f"💰 *Solde du portefeuille* : {initial_balance:,.2f} $\n\n"
    
    if len(holdings_before) > 0:
        telegram_msg += f"🔄 *Top 3 avant changement* : {', '.join(holdings_before)}\n"
    
    telegram_msg += f"📈 *Top 3 apres changement* : {', '.join(new_holdings) if new_holdings else 'Aucun'}\n\n"
    
    if not top_picks_details:
        telegram_msg += "❌ Aucun signal d'achat interessant aujourd'hui.\n"
    else:
        telegram_msg += "--- DETAILS DES NOUVEAUX ACHATS ---\n"
        for rank, pick in enumerate(top_picks_details, 1):
            ticker = pick['ticker']
            pred = pick['pred_return']
            nlp_score = pick['nlp_score']
            nlp_alert = pick['nlp_alert']
            
            val = investments.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            
            unrealized_pl = 0.0
            for pos in new_positions:
                if pos.symbol == ticker:
                    unrealized_pl = float(pos.unrealized_pl)
                    break
            
            # Injection pour l'export JSON
            pick['price'] = price
            pick['pred_return'] = pred * 100
            pick['meta_prob'] = meta_prob * 100
            pick['unrealized_pl'] = unrealized_pl
            pick['portfolio_pct'] = (val / initial_balance) * 100 if initial_balance > 0 else 0
            
            telegram_msg += (
                f"#{rank} *{ticker}*\n"
                f"   🔸 Rendement 5J prevu : +{pred*100:.2f}%\n"
                f"   🛡️ Confiance Meta-IA : {meta_prob*100:.1f}%\n"
                f"   🔸 Score Media : {nlp_alert} ({nlp_score:.2f})\n"
                f"   🔸 Montant investi : {val:,.2f} $\n"
                f"   🔸 Prix de l'action : {price:,.2f} $\n\n"
            )

    send_telegram_message(telegram_msg)
    print("Message Telegram envoye avec succes !")
    
    order_history = []
    completed_trades = []
    try:
        req_hist = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=200)
        orders = client.get_orders(filter=req_hist)
        
        filled_orders = [o for o in orders if o.filled_qty and float(o.filled_qty) > 0]
        filled_orders.sort(key=lambda x: x.filled_at) # Plus ancien au plus recent
        
        open_positions_tracking = {}
        
        for o in filled_orders:
            symbol = o.symbol
            side = str(o.side).split('.')[-1].upper()
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
            date_str = o.filled_at.strftime('%Y-%m-%d') if o.filled_at else ""
            
            if side == 'BUY':
                if symbol not in open_positions_tracking:
                    open_positions_tracking[symbol] = {'qty': 0, 'cost_basis': 0.0}
                prev_qty = open_positions_tracking[symbol]['qty']
                prev_cost = open_positions_tracking[symbol]['cost_basis']
                new_qty = prev_qty + qty
                new_cost = (prev_qty * prev_cost + qty * price) / new_qty if new_qty > 0 else price
                open_positions_tracking[symbol] = {'qty': new_qty, 'cost_basis': new_cost}
                
            elif side == 'SELL':
                if symbol in open_positions_tracking and open_positions_tracking[symbol]['qty'] > 0:
                    buy_price = open_positions_tracking[symbol]['cost_basis']
                    sell_qty = min(qty, open_positions_tracking[symbol]['qty'])
                    
                    realized_pl = (price - buy_price) * sell_qty
                    return_pct = ((price / buy_price) - 1) * 100 if buy_price > 0 else 0.0
                    
                    completed_trades.append({
                        'symbol': symbol,
                        'sell_date': date_str,
                        'buy_price': buy_price,
                        'sell_price': price,
                        'qty': sell_qty,
                        'realized_pl': realized_pl,
                        'return_pct': return_pct
                    })
                    
                    open_positions_tracking[symbol]['qty'] -= sell_qty
                    if open_positions_tracking[symbol]['qty'] <= 0.001:
                        del open_positions_tracking[symbol]
        
        recent_orders = sorted(filled_orders, key=lambda x: x.filled_at, reverse=True)[:20]
        for o in recent_orders:
            order_history.append({
                'symbol': o.symbol,
                'side': str(o.side).split('.')[-1],
                'qty': float(o.filled_qty),
                'price': float(o.filled_avg_price) if o.filled_avg_price else 0.0,
                'date': o.filled_at.strftime('%Y-%m-%d') if o.filled_at else ""
            })
            
    except Exception as e:
        print(f"Erreur recup historique : {e}")
        
    completed_trades.sort(key=lambda x: x['sell_date'], reverse=True)
        
    return {'balance': initial_balance, 'buying_power': float(account.buying_power), 'history': order_history, 'completed_trades': completed_trades}

if __name__ == '__main__':
    # Determine if last trading session was a Monday
    try:
        import yfinance as yf
        spy = yf.download('SPY', period='5d', progress=False)
        if not spy.empty:
            last_trading_day = spy.index[-1].weekday()
        else:
            last_trading_day = -1
    except Exception:
        last_trading_day = -1
        
    is_monday = (last_trading_day == 0)
    
    top_picks_details, full_universe_analysis = generate_todays_signals()
    account_info = execute_live_orders(top_picks_details, trade_today=is_monday)
    
    if account_info and client:
        perf_data = get_performance_comparison(client)
        dashboard_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'account': account_info,
            'top_picks': top_picks_details,
            'full_analysis': full_universe_analysis,
            'performance': perf_data
        }
        with open('dashboard_data.json', 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=4, ensure_ascii=False)
        print("\n[OK] Donnees JSON exportees pour l'Application Mobile QuickBooks.")
        
    print("Termine pour aujourd'hui ! Le robot a ferme ses portes.")
