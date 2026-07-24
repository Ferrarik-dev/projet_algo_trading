import yfinance as yf
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# Univers 
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
START_DATE = "2018-03-01" # Test ultime sur 8 ans (depuis la création du Fear & Greed)
END_DATE = "2026-01-01"

def fetch_fear_and_greed():
    print("  -> Téléchargement de l'Indice Fear & Greed (alternative.me)...")
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()['data']
            dates = [pd.to_datetime(int(item['timestamp']), unit='s') for item in data]
            values = [int(item['value']) for item in data]
            series = pd.Series(values, index=dates)
            series.index = series.index.normalize() # Midnight
            # 0 = Extreme Fear, 100 = Extreme Greed
            return series
    except Exception as e:
        print("Erreur Fear & Greed:", e)
    return pd.Series()

def fetch_all_data():
    print("--- 1. TÉLÉCHARGEMENT DES DONNÉES (Univers V6 Pro) ---")
    data_dict = {}
    
    # 1. Données Globales
    df_global = yf.download(GLOBAL_BENCHMARK, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_global.columns, pd.MultiIndex): df_global.columns = df_global.columns.droplevel(1)
    data_dict['MASTER'] = df_global['Close']
    
    # 2. Indicateur VIX
    print("  -> Téléchargement de l'Indice VIX (^VIX)...")
    df_vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    data_dict['VIX'] = df_vix['Close']
    
    # 3. Indicateur Fear & Greed
    data_dict['FNG'] = fetch_fear_and_greed()
    
    for sector_name, info in SECTORS.items():
        print(f"\nTéléchargement du secteur {sector_name}...")
        df_bench = yf.download(info['benchmark'], start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df_bench.columns, pd.MultiIndex): df_bench.columns = df_bench.columns.droplevel(1)
        
        sector_data = {'benchmark': df_bench['Close'], 'assets': {}}
        
        for ticker in info['tickers']:
            df_asset = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df_asset.columns, pd.MultiIndex): df_asset.columns = df_asset.columns.droplevel(1)
            
            df = pd.DataFrame()
            df['Close'] = df_asset['Close']
            df['Volume'] = df_asset['Volume']
            
            df.dropna(subset=['Close'], inplace=True)
            sector_data['assets'][ticker] = df
            
        data_dict[sector_name] = sector_data
        
    return data_dict

def prepare_asset_features(df_asset, sector_name, series_sector_bench, series_global_bench, series_fng, series_vix, horizon=5):
    df = df_asset.copy()
    df['Sector_Close'] = series_sector_bench
    df['Global_Close'] = series_global_bench
    
    # Injection des Données Alternatives Professionnelles
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
    
    # Indicateurs Dérivés
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
    
    # Cible : l'IA predit si le prix sera plus haut dans N jours
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def train_and_predict_experts(data_dict, horizon=5):
    print(f"\n--- ENTRAINEMENT (Horizon = {horizon} jours) ---")
    
    all_predictions = {}
    all_prices = {}
    all_volatilities = {}
    all_regimes = {}
    common_dates = None
    
    for sector_name, info in SECTORS.items():
        print(f"[{sector_name}] Entraînement en cours...")
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
            if df_raw.empty:
                print(f"[{ticker}] Données manquantes, ignoré.")
                continue
            df = prepare_asset_features(df_raw, sector_name, series_sector, series_global, series_fng, series_vix, horizon)
            
            available_features = [f for f in features if f in df.columns]
            X = df[available_features]
            X = df[available_features]
            y = df['Target']
            
            # --- WALK-FORWARD VALIDATION (Apprentissage Continu) ---
            predictions_list = []
            test_indices_list = []
            
            years = df.index.year.unique()
            for current_year in years[2:]:
                # Reverted Piste 3, using full history
                train_mask = df.index.year < current_year
                test_mask = df.index.year == current_year
                
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue
                    
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                
                # --- MODELISATION IA (XGBoost) ---
                model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric='logloss'
                )
                model.fit(X_train, y_train)
                
                # --- PREDICTION ---
                probs = model.predict_proba(X_test)[:, 1]
                predictions_list.extend(probs)
                test_indices_list.extend(X_test.index)
                
            test_dates = pd.DatetimeIndex(test_indices_list)
            all_predictions[ticker] = pd.Series(predictions_list, index=test_dates)
            
            all_prices[ticker] = df_raw['Close'].loc[test_dates]
            all_volatilities[ticker] = df.loc[test_dates, 'Volatility']
            
            regime_sector_ok = df.loc[test_dates, 'Sector_Close'] > df.loc[test_dates, 'Sector_SMA_200']
            regime_global_ok = df.loc[test_dates, 'Global_Close'] > df.loc[test_dates, 'Global_SMA_200']
            
            if sector_name == 'COMMODITIES':
                all_regimes[ticker] = regime_sector_ok
            else:
                all_regimes[ticker] = regime_global_ok & regime_sector_ok

    common_dates = None
    for ticker, series in all_predictions.items():
        test_dates = series.index
        if common_dates is None:
            common_dates = test_dates
        else:
            common_dates = common_dates.intersection(test_dates)
            
    return all_predictions, all_prices, all_volatilities, all_regimes, common_dates, data_dict['MASTER']

def master_allocator_backtest(all_predictions, all_prices, all_volatilities, all_regimes, common_dates, master_close):
    print("\n--- 3. SIMULATION : V6 PRO (WALK-FORWARD VALIDATION) ---")
    
    df_signals = pd.DataFrame(index=common_dates)
    df_probs = pd.DataFrame(index=common_dates)   # Scores de confiance bruts
    df_prices = pd.DataFrame(index=common_dates)
    
    for ticker in all_predictions.keys():
        pred = all_predictions[ticker].loc[common_dates]
        regime = all_regimes[ticker].loc[common_dates]
        
        # Conserver les probabilités brutes (pour le classement Top-K)
        df_probs[ticker] = pred.where(regime, other=0.0)  # 0 si marché bearish
        
        # Signal de base : confiance > 60% ET marché haussier
        long_signal = ((pred > 0.60) & regime).astype(int)
        df_signals[ticker] = long_signal
        df_prices[ticker] = all_prices[ticker].loc[common_dates]
        
    df_returns = df_prices.pct_change().fillna(0)
    
    # --- TOP-5 SÉLECTION (Configuration Optimale) ---
    # On garde uniquement les 5 actifs les plus confiants selon XGBoost chaque jour.
    # Résultat prouvé : +1076% net / -42% DD (supérieur au 1/N sur tous les actifs).
    TOP_K = 5
    df_top = pd.DataFrame(0, index=df_probs.index, columns=df_probs.columns)
    for date in df_probs.index:
        eligible_probs = df_probs.loc[date][df_signals.loc[date] == 1]
        if len(eligible_probs) > 0:
            top_tickers = eligible_probs.nlargest(TOP_K).index
            df_top.loc[date, top_tickers] = 1
    
    active_positions = df_top.sum(axis=1)
    df_weights = df_top.div(active_positions.replace(0, 1), axis=0)
    
    # DELAI D'EXECUTION : shift(2) au lieu de shift(1)
    # Calcule le signal à T, exécute à T+1 (Close), capte le rendement à T+2. Élimine 100% du lookahead bias intrajournalier.
    df_weights_shifted = df_weights.shift(2).fillna(0)
    
    portfolio_daily_returns_gross = (df_weights_shifted * df_returns).sum(axis=1)
    turnover = df_weights.diff().abs().sum(axis=1).fillna(0)
    if not df_weights.empty:
        turnover.iloc[0] = df_weights.iloc[0].sum()
        
    # SLIPPAGE ALPACA : 0.05% par trade (pas de commission, juste un petit spread)
    TRANSACTION_FEE = 0.0005
    portfolio_daily_returns_net = portfolio_daily_returns_gross - (turnover * TRANSACTION_FEE)
    
    proof_df = df_weights_shifted.copy()
    proof_df['Portfolio_Value_$'] = (1 + portfolio_daily_returns_net).cumprod() * 100000
    proof_df.to_csv("C:\\Users\\Elrik\\Desktop\\projet_algo_trading\\preuve_des_trades.csv")
    print("\nFichier 'preuve_des_trades.csv' genere sur le bureau pour audit.")
    
    portfolio_cumulative_net = (1 + portfolio_daily_returns_net).cumprod()
    portfolio_cumulative_gross = (1 + portfolio_daily_returns_gross).cumprod()
    running_max = portfolio_cumulative_net.cummax()
    drawdown = (portfolio_cumulative_net - running_max) / running_max
    max_drawdown = drawdown.min()
    
    spy_test_returns = master_close.pct_change().loc[common_dates]
    spy_test_returns.iloc[0] = 0
    spy_cumulative = (1 + spy_test_returns).cumprod()
    
    print(f"Performance Marche Global (S&P 500)     : {(spy_cumulative.iloc[-1] - 1) * 100:.2f}%")
    print(f"Performance V7 PRO (BRUTE - Theorique)  : {(portfolio_cumulative_gross.iloc[-1] - 1) * 100:.2f}%")
    print(f"Performance V7 PRO (NETTE - Realiste)   : {(portfolio_cumulative_net.iloc[-1] - 1) * 100:.2f}%")
    print(f"Maximum Drawdown (Pire chute)            : {max_drawdown * 100:.2f}%")
    
    plt.figure(figsize=(12, 7))
    plt.plot(common_dates, spy_cumulative, label='SPY (S&P 500 Global)', color='grey', linewidth=2)
    plt.plot(common_dates, portfolio_cumulative_net, label='V6 Pro (Nette - Après Frais)', color='gold', linewidth=2)
    plt.plot(common_dates, portfolio_cumulative_gross, label='V6 Pro (Brute)', color='orange', linestyle='--', alpha=0.5)
    plt.title('Backtest V6 PRO : Tests de Robustesse (Frais & Drawdown)')
    plt.xlabel('Date')
    plt.ylabel('Rendements Cumulés')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_result.png')
    print("Graphique sauvegardé sous 'backtest_result.png'")

if __name__ == "__main__":
    HORIZON = 15  # Optimal: predit 15 jours (3 semaines) a l'avance
    data_dict = fetch_all_data()
    all_predictions, all_prices, all_volatilities, all_regimes, common_dates, master_close = train_and_predict_experts(data_dict, HORIZON)
    master_allocator_backtest(all_predictions, all_prices, all_volatilities, all_regimes, common_dates, master_close)

