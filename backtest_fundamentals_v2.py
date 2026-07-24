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
INITIAL_CAPITAL = 10000.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'BAC', 'COST'
]

MACRO_TICKERS = {
    'VIX': '^VIX',        # Volatility Index
    'DXY': 'DX-Y.NYB',    # US Dollar Index
    'GOLD': 'GC=F',       # Gold Futures
    'SPY': 'SPY'          # S&P 500 ETF (pour le Beta)
}

def download_data():
    print("Telechargement des donnees (Actions + Macro)...")
    data = {}
    for ticker in UNIVERSE:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            data[ticker] = df
            
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            macro_data[name] = df
            
    return data, macro_data

def prepare_macro_features(macro_data):
    macro_features = pd.DataFrame()
    
    if 'VIX' in macro_data:
        macro_features['VIX_Ret_1d'] = macro_data['VIX']['Close'].pct_change(1)
    if 'DXY' in macro_data:
        macro_features['DXY_Ret_1d'] = macro_data['DXY']['Close'].pct_change(1)
    if 'GOLD' in macro_data:
        macro_features['GOLD_Ret_1d'] = macro_data['GOLD']['Close'].pct_change(1)
    if 'SPY' in macro_data:
        macro_features['SPY_Ret_1d'] = macro_data['SPY']['Close'].pct_change(1)
        macro_features['SPY_Var_63d'] = macro_features['SPY_Ret_1d'].rolling(63).var()
        
    macro_features.ffill(inplace=True) 
    return macro_features

def prepare_features(df_raw, macro_features):
    df = df_raw.copy()
    
    # Technical Features
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
    
    # --- CROSS-SECTIONAL MACRO FEATURES ---
    # Merge macro for calculation
    df = df.join(macro_features, how='left')
    df.ffill(inplace=True)
    
    # 63-day Rolling Correlations
    if 'VIX_Ret_1d' in df.columns:
        df['Corr_VIX_63d'] = df['Ret_1d'].rolling(63).corr(df['VIX_Ret_1d'])
    if 'DXY_Ret_1d' in df.columns:
        df['Corr_DXY_63d'] = df['Ret_1d'].rolling(63).corr(df['DXY_Ret_1d'])
    if 'GOLD_Ret_1d' in df.columns:
        df['Corr_GOLD_63d'] = df['Ret_1d'].rolling(63).corr(df['GOLD_Ret_1d'])
        
    # 63-day Rolling Beta vs SPY
    if 'SPY_Ret_1d' in df.columns and 'SPY_Var_63d' in df.columns:
        cov_spy = df['Ret_1d'].rolling(63).cov(df['SPY_Ret_1d'])
        df['Beta_SPY_63d'] = cov_spy / df['SPY_Var_63d']
    
    # Drop raw macro columns to avoid the "identical data" problem
    df.drop(columns=[col for col in macro_features.columns if col in df.columns], inplace=True)
    
    # Target
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df.dropna(inplace=True)
    return df

class RankSignalData(bt.feeds.PandasData):
    lines = ('predicted_ret',)
    params = (('predicted_ret', -1),)

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
            if len(data) > 0 and not np.isnan(data.predicted_ret[0]):
                candidates.append((data, data.predicted_ret[0]))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:self.params.top_n]
        top_candidates = [c for c in top_candidates if c[1] > 0]
        
        top_names = [c[0]._name for c in top_candidates]
        
        for data in self.datas:
            if self.getposition(data).size > 0 and data._name not in top_names:
                self.close(data=data)
                
        if len(top_candidates) > 0:
            target_pct = 0.95 / len(top_candidates)
            for data, pred_ret in top_candidates:
                self.order_target_percent(data=data, target=target_pct)

def run_simulation(top_n=3):
    print(f"\n--- PREPARATION DES DONNEES (SMART MACRO + TOP {top_n}) ---")
    data_dict, macro_data = download_data()
    
    macro_features = prepare_macro_features(macro_data)
    
    features_list = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month',
        'Corr_VIX_63d', 'Corr_DXY_63d', 'Corr_GOLD_63d', 'Beta_SPY_63d'
    ]
    
    all_features = {}
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df, macro_features)
        
    print("Génération des prédictions ML (Walk-Forward Fixe avec Smart Macro)...")
    trading_days = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='B')
    predictions_log = {}
    last_train_month = -1
    trained_model = None
    
    for current_date in trading_days:
        if current_date.month in [1, 4, 7, 10] and current_date.month != last_train_month:
            last_train_month = current_date.month
            
            X_train_list, y_train_list = [], []
            for ticker, df in all_features.items():
                df_hist = df[df.index < current_date]
                if len(df_hist) > 252 + 5:
                    df_hist = df_hist.iloc[:-5] # Purging
                    X_train_list.append(df_hist[features_list])
                    y_train_list.append(df_hist['Target_5d'])
                    
            if X_train_list:
                X_train = pd.concat(X_train_list)
                y_train = pd.concat(y_train_list)
                
                best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'num_leaves': 31,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                
                trained_model = lgb.LGBMRegressor(**best_params)
                trained_model.fit(X_train, y_train)
                
        if trained_model is None:
            continue
            
        for ticker, df in all_features.items():
            if ticker not in predictions_log: predictions_log[ticker] = {}
            if current_date in df.index:
                row = df.loc[current_date]
                if not row[features_list].isnull().any():
                    X_today = pd.DataFrame([row[features_list]])
                    pred = trained_model.predict(X_today)[0]
                    predictions_log[ticker][current_date] = pred

    print(f"\n--- LANCEMENT BACKTRADER (TOP {top_n}) ---")
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
                    
            df_bt = df_bt[['Open', 'High', 'Low', 'Close', 'Volume', 'predicted_ret']]
            data_feed = RankSignalData(dataname=df_bt, name=ticker)
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
    
    print(f"Resultats Top {top_n} avec SMART MACRO DATA :")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{abs_rendement:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%\n")
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + SMART MACRO DATA")
    import matplotlib.pyplot as plt
    strat_macro = run_simulation(top_n=3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat_macro.dates, strat_macro.portfolio_values, color='green', linewidth=2)
    plt.title('Performance du Bot : LightGBM + Smart Macro (Top 3)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(10000, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\macro_v2_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
