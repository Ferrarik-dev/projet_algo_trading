import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
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
    print("Telechargement des donnees (Actions + Macro pour Meta-Model)...")
    data = {}
    for ticker in UNIVERSE:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            data[ticker] = df
            
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
    
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.droplevel(1)
        
    return data, vix, spy

def prepare_features(df_raw, vix, spy):
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
    
    # Meta Features (Context)
    df['VIX_Close'] = vix['Close']
    df['SPY_Ret_21d'] = spy['Close'].pct_change(21)
    df.ffill(inplace=True)
    
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df.dropna(inplace=True)
    return df

class RankAndMetaSignalData(bt.feeds.PandasData):
    lines = ('predicted_ret', 'meta_prob')
    params = (('predicted_ret', -1), ('meta_prob', -1))

class StefanJansenMetaStrategy(bt.Strategy):
    params = (
        ('top_n', 3),
        ('meta_threshold', 0.50), # Le garde du corps exige au moins 50% de confiance pour valider
    )

    def __init__(self):
        self.portfolio_values = []
        self.dates = []

    def next(self):
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datetime.date(0))

        if self.datetime.date(0).weekday() != 0:
            return

        # 1. Le Modele Primaire propose son Top N
        candidates = []
        for data in self.datas:
            if len(data) > 0 and not np.isnan(data.predicted_ret[0]):
                candidates.append((data, data.predicted_ret[0], data.meta_prob[0]))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:self.params.top_n]
        top_candidates = [c for c in top_candidates if c[1] > 0]
        
        # 2. Le Garde du Corps (Meta Model) passe les candidats en revue
        approved_candidates = []
        for data, pred_ret, meta_prob in top_candidates:
            if meta_prob >= self.params.meta_threshold:
                approved_candidates.append(data)
            # Sinon, Veto ! On n'achete pas.
        
        approved_names = [c._name for c in approved_candidates]
        
        # Vente de tout ce qui n'est pas approuvé
        for data in self.datas:
            if self.getposition(data).size > 0 and data._name not in approved_names:
                self.close(data=data)
                
        # Achat du budget (Equally weighted sur les actions approuvees)
        if len(approved_candidates) > 0:
            target_pct = 0.95 / len(approved_candidates)
            for data in approved_candidates:
                self.order_target_percent(data=data, target=target_pct)

def run_simulation(top_n=3):
    print(f"\n--- PREPARATION DES DONNEES (META-LABELING + TOP {top_n}) ---")
    data_dict, vix, spy = download_data()
    
    primary_features = [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
    
    meta_features = [
        'Primary_Pred', 'Vol_21d', 'VIX_Close', 'SPY_Ret_21d'
    ]
    
    all_features = {}
    for ticker, df in data_dict.items():
        all_features[ticker] = prepare_features(df, vix, spy)
        
    print("Génération des prédictions (Modèle 1) et du Veto (Modèle 2)...")
    trading_days = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='B')
    predictions_log = {}
    last_train_month = -1
    
    model_primary = None
    model_meta = None
    
    for current_date in trading_days:
        if current_date.month in [1, 4, 7, 10] and current_date.month != last_train_month:
            last_train_month = current_date.month
            
            X_train_list, y_train_list = [], []
            df_hists = {}
            for ticker, df in all_features.items():
                df_hist = df[df.index < current_date]
                if len(df_hist) > 252 + 5:
                    df_hist = df_hist.iloc[:-5] # Purging
                    df_hists[ticker] = df_hist
                    X_train_list.append(df_hist[primary_features])
                    y_train_list.append(df_hist['Target_5d'])
                    
            if X_train_list:
                X_train_primary = pd.concat(X_train_list)
                y_train_primary = pd.concat(y_train_list)
                
                # --- ENTRAINEMENT MODELE 1 (LE CHERCHEUR) ---
                model_primary = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
                model_primary.fit(X_train_primary, y_train_primary)
                
                # --- GENERATION DES META-LABELS ---
                # On veut apprendre au Meta-Model : "Est-ce que le trade sera VRAIMENT positif ?"
                X_meta_list, y_meta_list = [], []
                for ticker, df_hist in df_hists.items():
                    preds = model_primary.predict(df_hist[primary_features])
                    
                    df_meta = pd.DataFrame(index=df_hist.index)
                    df_meta['Primary_Pred'] = preds
                    df_meta['Vol_21d'] = df_hist['Vol_21d']
                    df_meta['VIX_Close'] = df_hist['VIX_Close']
                    df_meta['SPY_Ret_21d'] = df_hist['SPY_Ret_21d']
                    
                    # Label binaire : 1 si le rendement futur a vraiment ete positif, 0 sinon
                    target_meta = (df_hist['Target_5d'] > 0).astype(int)
                    
                    X_meta_list.append(df_meta)
                    y_meta_list.append(target_meta)
                    
                X_train_meta = pd.concat(X_meta_list)
                y_train_meta = pd.concat(y_meta_list)
                
                # --- ENTRAINEMENT MODELE 2 (LE GARDE DU CORPS) ---
                # Un Random Forest pour etre robuste et different de LightGBM
                model_meta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                model_meta.fit(X_train_meta, y_train_meta)
                
        if model_primary is None or model_meta is None:
            continue
            
        for ticker, df in all_features.items():
            if ticker not in predictions_log: predictions_log[ticker] = {}
            if current_date in df.index:
                row = df.loc[current_date]
                if not row[primary_features].isnull().any() and not np.isnan(row['VIX_Close']):
                    # Modèle 1
                    X_prim_today = pd.DataFrame([row[primary_features]])
                    pred_ret = model_primary.predict(X_prim_today)[0]
                    
                    # Modèle 2
                    X_meta_today = pd.DataFrame([{
                        'Primary_Pred': pred_ret,
                        'Vol_21d': row['Vol_21d'],
                        'VIX_Close': row['VIX_Close'],
                        'SPY_Ret_21d': row['SPY_Ret_21d']
                    }])
                    # Probabilité que le trade soit gagnant (classe 1)
                    meta_prob = model_meta.predict_proba(X_meta_today)[0][1]
                    
                    predictions_log[ticker][current_date] = (pred_ret, meta_prob)

    print(f"\n--- LANCEMENT BACKTRADER (TOP {top_n} + VETO) ---")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=0.001)
    
    for ticker, df in all_features.items():
        if ticker in predictions_log:
            df_bt = df[df.index >= START_TRADING_DATE].copy()
            df_bt['predicted_ret'] = np.nan
            df_bt['meta_prob'] = np.nan
            
            for date, (pred, prob) in predictions_log[ticker].items():
                if date in df_bt.index:
                    df_bt.at[date, 'predicted_ret'] = pred
                    df_bt.at[date, 'meta_prob'] = prob
                    
            df_bt = df_bt[['Open', 'High', 'Low', 'Close', 'Volume', 'predicted_ret', 'meta_prob']]
            data_feed = RankAndMetaSignalData(dataname=df_bt, name=ticker)
            cerebro.adddata(data_feed)

    cerebro.addstrategy(StefanJansenMetaStrategy, top_n=top_n, meta_threshold=0.55)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    strat = results[0]
    rendement = strat.analyzers.returns.get_analysis()['rtot'] * 100
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    abs_rendement = (final_value / INITIAL_CAPITAL - 1) * 100
    
    print(f"Resultats Top {top_n} avec META-LABELING :")
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{abs_rendement:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%\n")
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + META-LABELING")
    import matplotlib.pyplot as plt
    strat_meta = run_simulation(top_n=3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat_meta.dates, strat_meta.portfolio_values, color='red', linewidth=2)
    plt.title('Performance du Bot : Modèle Primaire + Veto Meta-Labeling (Top 3)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(10000, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\metalabeling_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
