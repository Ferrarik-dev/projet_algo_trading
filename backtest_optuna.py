import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import backtrader as bt
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

START_DATE = '2016-01-01'
END_DATE = '2024-01-01'
START_TRADING_DATE = '2019-01-01'
INITIAL_CAPITAL = 10000.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'BAC', 'COST'
]

def download_data():
    print("Telechargement des donnees (2016-2024)...")
    data = {}
    for ticker in UNIVERSE:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            data[ticker] = df
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
    print(f"\n--- PREPARATION DES DONNEES (OPTUNA + TOP {top_n}) ---")
    data_dict = download_data()
    
    features_list = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
    
    all_features = {}
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df)
        
    print("Génération des prédictions ML (Walk-Forward avec LightGBM + Optuna)...")
    trading_days = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='B')
    predictions_log = {}
    last_train_month = -1
    trained_model = None
    
    for current_date in trading_days:
        # Entrainement et Optimisation chaque trimestre
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
                
                # --- OPTUNA STUDY ---
                def objective(trial):
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'max_depth': trial.suggest_int('max_depth', 3, 9),
                        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = []
                    
                    # On trie d'abord par date pour que le TimeSeriesSplit ait du sens (les donnees sont melangees par ticker actuellement)
                    # Comme l'index est la date, on s'assure qu'il est trie
                    X_sorted = X_train.sort_index()
                    y_sorted = y_train.loc[X_sorted.index]
                    
                    for train_idx, val_idx in tscv.split(X_sorted):
                        X_t, X_v = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
                        y_t, y_v = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]
                        
                        m = lgb.LGBMRegressor(**param)
                        m.fit(X_t, y_t)
                        preds = m.predict(X_v)
                        scores.append(mean_squared_error(y_v, preds))
                        
                    return np.mean(scores)
                
                print(f"[{current_date.date()}] Optimisation Optuna en cours (20 essais)...")
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=20)
                
                best_params = study.best_params
                best_params['random_state'] = 42
                best_params['n_jobs'] = -1
                best_params['verbose'] = -1
                
                trained_model = lgb.LGBMRegressor(**best_params)
                trained_model.fit(X_train, y_train)
                
        # Prediction chaque jour
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
    
    print(f"Resultats Top {top_n} avec Optuna :")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{abs_rendement:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%\n")
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + OPTUNA")
    import matplotlib.pyplot as plt
    strat_optuna = run_simulation(top_n=3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat_optuna.dates, strat_optuna.portfolio_values, color='purple', linewidth=2)
    plt.title('Performance du Bot : LightGBM + Optuna (Top 3)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(10000, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\optuna_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
