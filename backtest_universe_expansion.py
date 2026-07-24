import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import backtrader as bt
import warnings
warnings.filterwarnings('ignore')

START_DATE = '2016-01-01'
END_DATE = '2024-01-01'
START_TRADING_DATE = '2019-01-01'
INITIAL_CAPITAL = 1000000.0 # On passe à 1M$ pour pouvoir acheter des actions très chères sans fractionnement

# 95 Actions hyper-liquides du NASDAQ
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP', 'COST', 
    'CSCO', 'TMUS', 'ADBE', 'TXN', 'NFLX', 'AMD', 'CMCSA', 'INTU', 'QCOM', 'AMGN', 
    'HON', 'INTC', 'ISRG', 'IBM', 'GILD', 'BKNG', 'VRTX', 'SBUX', 'MDLZ', 'REGN', 
    'ADI', 'PANW', 'ADP', 'LRCX', 'MU', 'KLAC', 'SNPS', 'MELI', 'CDNS', 'CSX', 
    'PYPL', 'MAR', 'ABNB', 'ORLY', 'CTAS', 'MNST', 'WDAY', 'PCAR', 'NXPI', 'KDP', 
    'CRWD', 'MCHP', 'DXCM', 'ASML', 'LULU', 'FTNT', 'EXC', 'MRVL', 'KHC', 'PAYX', 
    'IDXX', 'ODFL', 'ROST', 'CTSH', 'BIIB', 'EA', 'BKR', 'FAST', 'GEHC', 'CSGP', 
    'TTWO', 'VRSK', 'DLTR', 'WBA', 'ANSS', 'ALGN', 'EBAY', 'ILMN', 'SIRI', 'ZM', 
    'DOCU', 'SPLK', 'OKTA', 'SWKS', 'DDOG', 'TEAM', 'ZS', 'MDB', 'CRSP', 'ENPH',
    'JPM', 'V', 'WMT', 'JNJ', 'PG'
]

def download_data():
    print(f"Telechargement en masse de {len(UNIVERSE)} actions (Parallel)...")
    # Téléchargement optimisé en 1 seule requête
    df_all = yf.download(tickers=" ".join(UNIVERSE), start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
    
    data = {}
    for ticker in UNIVERSE:
        if ticker in df_all.columns.levels[0]:
            df_ticker = df_all[ticker].copy()
            df_ticker.dropna(inplace=True) # Supprime les jours sans données (ex: avant IPO)
            # On ne garde que les actions qui ont existé pendant toute la période (8 ans = ~2000 jours)
            if not df_ticker.empty and len(df_ticker) > 1800:
                data[ticker] = df_ticker
                
    print(f"Téléchargement réussi pour {len(data)} actions.")
    return data

def prepare_features(df_raw):
    df = df_raw.copy()
    
    for lag in [1, 5, 21, 42, 63, 126, 252]:
        df[f'Ret_{lag}d'] = df['Close'].pct_change(lag)
        
    df['Vol_21d'] = df['Ret_1d'].rolling(21).std()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['Dist_BB_Upper'] = (df['Close'] / df['BB_Upper']) - 1
    df['Dist_BB_Lower'] = (df['Close'] / df['BB_Lower']) - 1
    
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    # Filtre de Tendance Macro (SMA 200)
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Is_Uptrend'] = (df['Close'] > df['SMA_200']).astype(int)
    
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df.dropna(inplace=True)
    return df

class RankSignalData(bt.feeds.PandasData):
    lines = ('predicted_ret', 'is_uptrend',)
    params = (
        ('predicted_ret', -1),
        ('is_uptrend', -1),
    )

class StefanJansenRankingStrategy(bt.Strategy):
    params = (('top_n', 3),)

    def __init__(self):
        self.portfolio_values = []
        self.dates = []

    def next(self):
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datetime.date(0))

        if self.datetime.date(0).weekday() != 0:
            return

        candidates = []
        for data in self.datas:
            # On vérifie que la prédiction existe ET que l'action est en Uptrend (Prix > SMA 200)
            if len(data) > 0 and not np.isnan(data.predicted_ret[0]) and data.is_uptrend[0] == 1:
                candidates.append((data, data.predicted_ret[0]))
                
        print(f"[{self.datetime.date(0)}] Candidates found: {len(candidates)}")
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:self.params.top_n]
        top_candidates = [c for c in top_candidates if c[1] > 0]
        
        if top_candidates:
            print(f"[{self.datetime.date(0)}] Executing orders for top {len(top_candidates)}...")
        
        top_names = [c[0]._name for c in top_candidates]
        
        for data in self.datas:
            if self.getposition(data).size > 0 and data._name not in top_names:
                self.close(data=data)
                
        if len(top_candidates) > 0:
            target_pct = 0.95 / len(top_candidates)
            for data, pred_ret in top_candidates:
                self.order_target_percent(data=data, target=target_pct)

def run_simulation(top_n=3):
    print(f"\n--- PREPARATION DES DONNEES (NASDAQ 100 - TOP {top_n}) ---")
    data_dict = download_data()
    
    features_list = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
    
    all_features = {}
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df)
        
    print(f"Génération des prédictions (Batch par trimestre pour {len(all_features)} actions)...")
    
    predictions_log = {}
    dates_index = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='QE')
    
    for i in range(len(dates_index) - 1):
        quarter_start = dates_index[i]
        quarter_end = dates_index[i+1]
        
        X_train_list, y_train_list = [], []
        X_test_dict = {}
        
        for ticker, df in all_features.items():
            df_hist = df[df.index < quarter_start]
            if len(df_hist) > 252 + 5:
                df_hist = df_hist.iloc[:-5] # Purging
                X_train_list.append(df_hist[features_list])
                y_train_list.append(df_hist['Target_5d'])
                
            df_test = df[(df.index >= quarter_start) & (df.index < quarter_end)]
            if not df_test.empty:
                X_test_dict[ticker] = (df_test.index, df_test[features_list])
                
        if X_train_list and X_test_dict:
            X_train = pd.concat(X_train_list)
            y_train = pd.concat(y_train_list)
            
            # Paramètres fixes du "Baseline"
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            
            for ticker, (indices, X_test) in X_test_dict.items():
                if ticker not in predictions_log: predictions_log[ticker] = {}
                preds = model.predict(X_test)
                for date, pred in zip(indices, preds):
                    predictions_log[ticker][date] = pred

    print(f"\n--- LANCEMENT BACKTRADER (TOP {top_n} sur {len(all_features)} actions) ---")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=0.001)
    
    for ticker, df in all_features.items():
        if ticker in predictions_log:
            df_bt = df[df.index >= START_TRADING_DATE].copy()
            df_bt['predicted_ret'] = np.nan
            
            for date, pred in predictions_log[ticker].items():
                if date in df_bt.index:
                    df_bt.at[date, 'predicted_ret'] = pred
                    
            df_bt = df_bt[['Open', 'High', 'Low', 'Close', 'Volume', 'predicted_ret', 'Is_Uptrend']]
            data_feed = RankSignalData(dataname=df_bt, name=ticker, is_uptrend='Is_Uptrend')
            cerebro.adddata(data_feed)

    cerebro.addstrategy(StefanJansenRankingStrategy, top_n=top_n)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    strat = results[0]
    rendement = strat.analyzers.returns.get_analysis()['rtot'] * 100
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    abs_rendement = (final_value / INITIAL_CAPITAL - 1) * 100
    
    print(f"Resultats Top {top_n} avec NASDAQ 100 :")
    print(f"Total days simulated: {len(strat.dates)}")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{abs_rendement:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%\n")
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + UNIVERS NASDAQ 100 + SMA200")
    import matplotlib.pyplot as plt
    strat_100 = run_simulation(top_n=3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat_100.dates, strat_100.portfolio_values, color='gold', linewidth=2)
    plt.title('Performance du Bot : 100 Actions NASDAQ + SMA200 (Top 3)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\nasdaq_sma200_top3_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
