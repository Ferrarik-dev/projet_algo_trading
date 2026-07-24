import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

def download_data():
    print("Telechargement des donnees (Actions)...")
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
    print(f"\n--- PREPARATION DES DONNEES (ENSEMBLE MODELS + TOP {top_n}) ---")
    data_dict = download_data()
    
    features_list = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
    
    all_features = {}
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df)
        
    print("Génération des prédictions (Comité de 3 Modèles)...")
    
    predictions_log = {}
    
    # Construction des quarters (trimestres)
    dates_index = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='QE')
    
    for i in range(len(dates_index) - 1):
        quarter_start = dates_index[i]
        quarter_end = dates_index[i+1]
        
        print(f"Entraînement Walk-Forward : Test sur {quarter_start.date()} au {quarter_end.date()}")
        
        X_train_list, y_train_list = [], []
        X_test_dict = {}
        
        for ticker, df in all_features.items():
            # Train Data : Tout avant le trimestre actuel (avec purge)
            df_hist = df[df.index < quarter_start]
            if len(df_hist) > 252 + 5:
                df_hist = df_hist.iloc[:-5] # Purging
                X_train_list.append(df_hist[features_list])
                y_train_list.append(df_hist['Target_5d'])
                
            # Test Data : Le trimestre actuel
            df_test = df[(df.index >= quarter_start) & (df.index < quarter_end)]
            if not df_test.empty:
                X_test_dict[ticker] = (df_test.index, df_test[features_list])
                
        if X_train_list and X_test_dict:
            X_train = pd.concat(X_train_list)
            y_train = pd.concat(y_train_list)
            
            # --- MODEL 1 : LightGBM ---
            model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
            model_lgb.fit(X_train, y_train)
            
            # --- MODEL 2 : Random Forest ---
            model_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model_rf.fit(X_train, y_train)
            
            # --- MODEL 3 : Ridge Regression (avec StandardScaler) ---
            model_ridge = make_pipeline(StandardScaler(), RidgeCV())
            model_ridge.fit(X_train, y_train)
            
            # BATCH INFERENCE
            for ticker, (indices, X_test) in X_test_dict.items():
                if ticker not in predictions_log: predictions_log[ticker] = {}
                
                pred_lgb = model_lgb.predict(X_test)
                pred_rf = model_rf.predict(X_test)
                pred_ridge = model_ridge.predict(X_test)
                
                # Le Super Score (Moyenne des 3)
                ensemble_preds = (pred_lgb + pred_rf + pred_ridge) / 3.0
                
                for date, ensemble_pred in zip(indices, ensemble_preds):
                    predictions_log[ticker][date] = ensemble_pred

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
    
    print(f"Resultats Top {top_n} avec ENSEMBLE (3 Modeles) :")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{abs_rendement:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%\n")
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + ENSEMBLE LEARNING")
    import matplotlib.pyplot as plt
    strat_ensemble = run_simulation(top_n=3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat_ensemble.dates, strat_ensemble.portfolio_values, color='purple', linewidth=2)
    plt.title('Performance du Bot : Ensemble (LGBM + RF + Ridge) (Top 3)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(10000, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\ensemble_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
