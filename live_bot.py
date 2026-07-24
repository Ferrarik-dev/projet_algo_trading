import yfinance as yf
import pandas as pd
import xgboost as xgb
import requests
import warnings
import os
import datetime
import numpy as np
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

warnings.filterwarnings('ignore')

# ==========================================
# 1. INITIALISATION ET SÉCURITÉ
# ==========================================
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

print("================================================")
print("   ROBOT DE TRADING V7 PRO (LIVE MODE)   ")
print("================================================")

if not SECRET_KEY or SECRET_KEY == "VOTRE_CLEF_SECRETE_ICI":
    print("ERREUR CRITIQUE : La clé secrète Alpaca est manquante dans le fichier .env !")
    print("Veuillez coller votre Secret Key dans le fichier .env avant de lancer le robot.")
    exit(1)

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

try:
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    print(f"Connexion Alpaca réussie ! Solde du compte : {portfolio_value:.2f} $")
except Exception as e:
    print(f"Erreur de connexion à Alpaca : {e}")
    exit(1)

# ==========================================
# 2. UNIVERS ET PARAMÈTRES
# ==========================================
SECTORS = {
    'CRYPTO': {
        'tickers': ['ETH-USD', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'LTC-USD'],
        'benchmark': 'BTC-USD',
    },
    'MIDCAPS': {
        'tickers': ['ENPH', 'OKTA', 'CCJ', 'MDB', 'SHOP'],
        'benchmark': 'IWM',
    },
    'COMMODITIES': {
        'tickers': ['GLD', 'SLV', 'USO', 'WEAT', 'CPER'],
        'benchmark': 'DBC',
    }
}
GLOBAL_BENCHMARK = 'SPY' 
START_DATE = "2018-03-01" 
# On télécharge les données jusqu'à aujourd'hui
TODAY_STR = datetime.datetime.now().strftime('%Y-%m-%d')

# Pour Alpaca, les symboles Crypto utilisent un format spécial (ex: ETH/USD)
# On fera la conversion au moment de l'ordre.

def fetch_fear_and_greed():
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()['data']
            dates = [pd.to_datetime(int(item['timestamp']), unit='s') for item in data]
            values = [int(item['value']) for item in data]
            series = pd.Series(values, index=dates)
            series.index = series.index.normalize() 
            return series
    except Exception as e:
        pass
    return pd.Series()

def fetch_all_data():
    print("\nEtape 1 : Telechargement des donnees du marche en direct...")
    data_dict = {}
    
    df_global = yf.download(GLOBAL_BENCHMARK, start=START_DATE, end=TODAY_STR, progress=False)
    if isinstance(df_global.columns, pd.MultiIndex): df_global.columns = df_global.columns.droplevel(1)
    data_dict['MASTER'] = df_global['Close']
    
    df_vix = yf.download('^VIX', start=START_DATE, end=TODAY_STR, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    data_dict['VIX'] = df_vix['Close']
    
    data_dict['FNG'] = fetch_fear_and_greed()
    
    for sector_name, info in SECTORS.items():
        df_bench = yf.download(info['benchmark'], start=START_DATE, end=TODAY_STR, progress=False)
        if isinstance(df_bench.columns, pd.MultiIndex): df_bench.columns = df_bench.columns.droplevel(1)
        
        sector_data = {'benchmark': df_bench['Close'], 'assets': {}}
        for ticker in info['tickers']:
            df_asset = yf.download(ticker, start=START_DATE, end=TODAY_STR, progress=False)
            if isinstance(df_asset.columns, pd.MultiIndex): df_asset.columns = df_asset.columns.droplevel(1)
            
            df = pd.DataFrame()
            df['Close'] = df_asset['Close']
            df['Volume'] = df_asset['Volume']
            df.dropna(subset=['Close'], inplace=True)
            sector_data['assets'][ticker] = df
            
        data_dict[sector_name] = sector_data
        
    return data_dict

def prepare_asset_features(df_asset, sector_name, series_sector_bench, series_global_bench, series_fng, series_vix):
    df = df_asset.copy()
    df['Sector_Close'] = series_sector_bench
    df['Global_Close'] = series_global_bench
    
    if sector_name == 'CRYPTO' and not series_fng.empty:
        df['Fear_Greed'] = series_fng
        df['Fear_Greed'] = df['Fear_Greed'].ffill()
    
    if sector_name != 'CRYPTO' and not series_vix.empty:
        df['VIX'] = series_vix
        df['VIX'] = df['VIX'].ffill()
        
    df.dropna(subset=['Sector_Close', 'Global_Close'], inplace=True)
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    df['Sector_Returns'] = df['Sector_Close'].pct_change()
    df['Sector_SMA_200'] = df['Sector_Close'].rolling(window=200).mean()
    df['Global_SMA_200'] = df['Global_Close'].rolling(window=200).mean()
    
    if 'VIX' in df.columns:
        df['VIX_Ratio'] = df['VIX'] / df['VIX'].rolling(window=30).mean()
    
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Relative_Strength_Sector'] = df['Close'].pct_change(periods=10) - df['Sector_Close'].pct_change(periods=10)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # La cible pour l'entraînement historique (Horizon = 15 jours V7 PRO)
    df['Target'] = (df['Close'].shift(-15) > df['Close']).astype(int)
    
    return df

def generate_todays_signals(data_dict):
    print("\nEtape 2 : Entrainement des IA et Prediction pour Aujourd'hui...")
    scored_signals = []
    
    for sector_name, info in SECTORS.items():
        series_sector = data_dict[sector_name]['benchmark']
        series_global = data_dict['MASTER']
        series_fng = data_dict['FNG']
        series_vix = data_dict['VIX']
        
        features = ['SMA_10', 'SMA_50', 'Returns', 'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector']
        if sector_name == 'CRYPTO':
            features.append('Fear_Greed')
        else:
            features.append('VIX')
            features.append('VIX_Ratio')
            
        for ticker, df_raw in data_dict[sector_name]['assets'].items():
            df = prepare_asset_features(df_raw, sector_name, series_sector, series_global, series_fng, series_vix)
            
            # Séparer l'historique (pour entraîner) de la toute dernière ligne (Aujourd'hui, pour prédire)
            df_train = df.iloc[:-1].dropna() # On enlève la dernière ligne car elle n'a pas de Target valide (shift -15)
            df_today = df.iloc[-1:] # La donnée du jour
            
            # S'il manque des données pour aujourd'hui, on ignore
            if df_today[features].isnull().values.any():
                continue
                
            X_train = df_train[features]
            y_train = df_train['Target']
            X_today = df_today[features]
            
            # Apprentissage Continu (Walk-Forward) : L'IA s'entraîne sur TOUTE l'histoire jusqu'à hier
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            
            # Prédiction pour AUJOURD'HUI
            prob_today = model.predict_proba(X_today)[:, 1][0]
            
            # Vérification du Régime (Moyennes mobiles 200 jours)
            regime_sector_ok = df_today['Sector_Close'].values[0] > df_today['Sector_SMA_200'].values[0]
            regime_global_ok = df_today['Global_Close'].values[0] > df_today['Global_SMA_200'].values[0]
            
            if sector_name == 'COMMODITIES':
                regime_ok = regime_sector_ok
            else:
                regime_ok = regime_global_ok and regime_sector_ok
                
            print(f"[{ticker}] Probabilité de Hausse: {prob_today*100:.1f}% | Régime OK: {regime_ok}")
            
            if prob_today > 0.60 and regime_ok:
                scored_signals.append((ticker, prob_today))
                
    # V7 PRO : Trier par probabilité et ne garder que le Top-5 absolu
    scored_signals.sort(key=lambda x: x[1], reverse=True)
    top_5 = [item[0] for item in scored_signals[:5]]
    
    print(f"\n--- TOP-5 DU JOUR (V7 PRO) ---")
    for i, item in enumerate(scored_signals[:5]):
        print(f"{i+1}. {item[0]} ({item[1]*100:.1f}%)")
        
    return top_5

def format_ticker_for_alpaca(ticker):
    # Convertit 'ETH-USD' en 'ETHUSD' (ou 'ETH/USD' selon l'API crypto, Alpaca V2 utilise souvent 'ETH/USD')
    if '-USD' in ticker:
        return ticker.replace('-USD', '/USD')
    return ticker

def execute_live_orders(buy_signals):
    print("\nEtape 3 : Execution des Ordres sur Alpaca...")
    
    # 1. Lister les positions actuelles
    positions = client.get_all_positions()
    current_holdings = {pos.symbol: float(pos.qty) for pos in positions}
    
    print(f"Positions actuellement detenues : {list(current_holdings.keys())}")
    print(f"Nouveaux Signaux d'Achat generes par l'IA : {buy_signals}")
    
    alpaca_buy_signals = [format_ticker_for_alpaca(t) for t in buy_signals]
    
    # 2. Vendre ce qui n'a plus de signal
    for symbol in current_holdings.keys():
        if symbol not in alpaca_buy_signals:
            print(f"Vente de {symbol} (Sorti du Top-5)")
            client.close_position(symbol)
            
    # 3. Acheter les nouveaux signaux
    if len(alpaca_buy_signals) == 0:
        print("Aucun signal d'achat aujourd'hui. L'IA reste en sécurité (Cash).")
        return
        
    # Smart Rebalance V7 PRO : Identifier les NOUVEAUX actifs à acheter
    new_assets = [s for s in alpaca_buy_signals if s not in current_holdings]
    
    if len(new_assets) > 0:
        account = client.get_account()
        buying_power = float(account.buying_power)
        # On utilise 95% du cash disponible, divisé par le nombre de NOUVEAUX actifs à financer
        budget_per_asset = (buying_power * 0.95) / len(new_assets)
        budget_per_asset = round(budget_per_asset, 2)
        print(f"Budget alloué par NOUVEL actif : {budget_per_asset} $")
    else:
        print("Tous les actifs du Top-5 sont déjà en portefeuille. Aucun nouvel achat nécessaire.")
        budget_per_asset = 0
    
    print(f"Budget alloué par actif : {budget_per_asset} $")
    
    for symbol in new_assets:
        if symbol not in current_holdings:
            print(f"Achat de {symbol}")
            try:
                # Ordre Notionnel (Basé sur le budget en $, fractionnel automatique)
                req = MarketOrderRequest(
                    symbol=symbol,
                    notional=budget_per_asset,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                client.submit_order(req)
                print(f" -> Ordre exécuté : {budget_per_asset:.2f} $ de {symbol}")
            except Exception as e:
                print(f"Erreur lors de l'achat de {symbol} : {e}")
        else:
            print(f"On conserve deja {symbol}, aucun nouvel ordre.")

if __name__ == "__main__":
    data_dict = fetch_all_data()
    buy_signals = generate_todays_signals(data_dict)
    execute_live_orders(buy_signals)
    print("\nTermine pour aujourd'hui ! Le robot a ferme ses portes.")
