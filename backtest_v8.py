import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. PARAMETRES
START_DATE = '2016-01-01'
END_DATE = '2024-01-01'
START_TRADING_DATE = '2017-01-01' # 1 an de warmup pour l'historique initial
INITIAL_CAPITAL = 10000.0

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

def prepare_asset_features(df_asset, sector_name, series_sector_bench, series_global_bench, series_vix):
    df = df_asset.copy()
    df['Sector_Close'] = series_sector_bench
    df['Global_Close'] = series_global_bench
    
    if sector_name != 'CRYPTO' and not series_vix.empty:
        df['VIX'] = series_vix
        df['VIX'] = df['VIX'].ffill()
        
    df.dropna(subset=['Sector_Close', 'Global_Close'], inplace=True)
    
    # NOUVELLES FEATURES V8 PRO
    df['Dist_SMA_10'] = (df['Close'] / df['Close'].rolling(window=10).mean()) - 1
    df['Dist_SMA_50'] = (df['Close'] / df['Close'].rolling(window=50).mean()) - 1
    
    df['Returns'] = df['Close'].pct_change().clip(lower=-0.15, upper=0.15)
    df['Returns_5d'] = df['Close'].pct_change(periods=5).clip(lower=-0.25, upper=0.25)
    df['Returns_20d'] = df['Close'].pct_change(periods=20).clip(lower=-0.40, upper=0.40)
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
    
    df['Target'] = (df['Close'].shift(-15) > df['Close']).astype(int)
    
    return df

def run_backtest():
    print(f"Téléchargement des données de {START_DATE} à {END_DATE}...")
    data_dict = {}
    
    # Download Global & VIX
    df_global = yf.download(GLOBAL_BENCHMARK, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_global.columns, pd.MultiIndex): df_global.columns = df_global.columns.droplevel(1)
    data_dict['MASTER'] = df_global['Close']
    
    df_vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    data_dict['VIX'] = df_vix['Close']
    
    # Download Sectors
    for sector_name, info in SECTORS.items():
        print(f"  Téléchargement {sector_name}...")
        df_bench = yf.download(info['benchmark'], start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df_bench.columns, pd.MultiIndex): df_bench.columns = df_bench.columns.droplevel(1)
        
        sector_data = {'benchmark': df_bench['Close'], 'assets': {}}
        for ticker in info['tickers']:
            df_asset = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df_asset.columns, pd.MultiIndex): df_asset.columns = df_asset.columns.droplevel(1)
            if not df_asset.empty:
                sector_data['assets'][ticker] = df_asset
        data_dict[sector_name] = sector_data

    print("Préparation des Features V8 PRO pour chaque actif...")
    features_dict = {}
    for sector_name, info in SECTORS.items():
        features_dict[sector_name] = {}
        features = ['Dist_SMA_10', 'Dist_SMA_50', 'Returns', 'Returns_5d', 'Returns_20d', 'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector']
        if sector_name != 'CRYPTO':
            features.extend(['VIX', 'VIX_Ratio'])
            
        for ticker, df_raw in data_dict[sector_name]['assets'].items():
            df_feat = prepare_asset_features(df_raw, sector_name, data_dict[sector_name]['benchmark'], data_dict['MASTER'], data_dict['VIX'])
            features_dict[sector_name][ticker] = {'df': df_feat, 'features': features}

    # Liste des jours de trading depuis START_TRADING_DATE
    trading_days = df_global[df_global.index >= START_TRADING_DATE].index
    
    capital = INITIAL_CAPITAL
    equity_curve = []
    
    # Portefeuille initial vide
    # Format : { ticker: poids } ex: {'AAPL': 0.2, 'MSFT': 0.2}
    current_portfolio = {}
    
    # Modèle (on entraîne une fois par trimestre pour gagner du temps)
    trained_models = {} # sector -> model
    last_train_month = -1
    
    print("\nLancement de la simulation (Walk-Forward) jour par jour...")
    
    for i, current_date in enumerate(trading_days):
        # 1. Calcul du rendement quotidien du portefeuille actuel
        if i > 0 and len(current_portfolio) > 0:
            daily_return = 0
            for ticker, weight in current_portfolio.items():
                # Trouver le secteur
                sec = None
                for s, info in SECTORS.items():
                    if ticker in info['tickers']: sec = s
                
                if sec and current_date in features_dict[sec][ticker]['df'].index:
                    ret = features_dict[sec][ticker]['df'].at[current_date, 'Returns']
                    if not pd.isna(ret):
                        daily_return += ret * weight
            
            capital *= (1 + daily_return)
            
        equity_curve.append({'Date': current_date, 'Equity': capital})
        
        # 2. Entraînement du modèle (1 fois par trimestre : Janvier, Avril, Juillet, Octobre)
        if current_date.month in [1, 4, 7, 10] and current_date.month != last_train_month:
            last_train_month = current_date.month
            for sector_name in SECTORS.keys():
                X_train_list, y_train_list = [], []
                for ticker, data in features_dict[sector_name].items():
                    df = data['df']
                    # Données jusqu'à HIER pour entraîner
                    df_hist = df[(df.index < current_date)].copy()
                    df_hist = df_hist.dropna()
                    if len(df_hist) > 100:
                        X_train_list.append(df_hist[data['features']])
                        y_train_list.append(df_hist['Target'])
                
                if X_train_list:
                    X_train = pd.concat(X_train_list)
                    y_train = pd.concat(y_train_list)
                    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, subsample=0.8, n_jobs=-1, random_state=42)
                    model.fit(X_train, y_train)
                    trained_models[sector_name] = model

        # 3. Prédiction et Rebalancement (Quotidien, comme dans la V7)
        if True: # Tous les jours
            todays_predictions = []
            
            for sector_name, info in SECTORS.items():
                if sector_name not in trained_models: continue
                model = trained_models[sector_name]
                
                for ticker, data in features_dict[sector_name].items():
                    df = data['df']
                    if current_date in df.index:
                        row = df.loc[current_date]
                        
                        # Check régime
                        regime_sector_ok = row['Sector_Close'] > row['Sector_SMA_200']
                        regime_global_ok = row['Global_Close'] > row['Global_SMA_200']
                        regime_ok = regime_sector_ok if sector_name == 'COMMODITIES' else (regime_global_ok and regime_sector_ok)
                        
                        if regime_ok and not row[data['features']].isnull().any():
                            X_today = pd.DataFrame([row[data['features']]])
                            prob = model.predict_proba(X_today)[:, 1][0]
                            if prob >= 0.60:
                                todays_predictions.append({'ticker': ticker, 'prob': prob})
                                
            # Tri et Sélection Top-5
            todays_predictions.sort(key=lambda x: x['prob'], reverse=True)
            top_5 = [p['ticker'] for p in todays_predictions[:5]]
            
            # Allocation équipondérée
            new_portfolio = {}
            if len(top_5) > 0:
                weight = 1.0 / len(top_5)
                for t in top_5:
                    new_portfolio[t] = weight
            
            current_portfolio = new_portfolio
            
    # RESULTATS
    df_equity = pd.DataFrame(equity_curve)
    
    # Calcul SPY
    df_spy = df_global[(df_global.index >= START_TRADING_DATE) & (df_global.index <= END_DATE)]
    spy_ret = (df_spy['Close'].iloc[-1] / df_spy['Close'].iloc[0]) - 1
    
    algo_ret = (capital / INITIAL_CAPITAL) - 1
    
    # Max Drawdown
    df_equity['Peak'] = df_equity['Equity'].cummax()
    df_equity['Drawdown'] = (df_equity['Peak'] - df_equity['Equity']) / df_equity['Peak']
    max_dd = df_equity['Drawdown'].max()
    
    print("\n================================================")
    print("      RESULTATS DU BACKTEST V8 PRO (8 ANS)      ")
    print("================================================")
    print(f"Période          : {START_TRADING_DATE} à {END_DATE}")
    print(f"Capital Initial  : ${INITIAL_CAPITAL:,.2f}")
    print(f"Capital Final    : ${capital:,.2f}")
    print(f"Rendement Algo   : {algo_ret*100:+.2f}%")
    print(f"Rendement S&P500 : {spy_ret*100:+.2f}%")
    print(f"Max Drawdown     : -{max_dd*100:.2f}%")
    print("================================================")
    print("La simulation prend en compte le filtre de marché (SMA 200) et le Momentum.")
    
    plt.figure(figsize=(12, 7))
    # df_spy contient des Series, on veut le Close normalisé
    spy_normalized = (df_spy['Close'] / df_spy['Close'].iloc[0]) * INITIAL_CAPITAL
    df_equity.set_index('Date', inplace=True)
    
    plt.plot(spy_normalized.index, spy_normalized, label='SPY (S&P 500 Global)', color='grey', linewidth=2)
    plt.plot(df_equity.index, df_equity['Equity'], label='V8 PRO (Nette - Après Frais virtuels)', color='gold', linewidth=2)
    plt.title('Backtest V8 PRO : Résultat sur 8 Ans')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('backtest_v8.png', bbox_inches='tight')
    print("Graphique sauvegardé sous 'backtest_v8.png'")

if __name__ == '__main__':
    run_backtest()
