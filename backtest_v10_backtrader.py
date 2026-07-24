import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import backtrader as bt
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

START_DATE = '2016-01-01'
END_DATE = '2024-01-01'
START_TRADING_DATE = '2017-01-01'
INITIAL_CAPITAL = 10000.0

SECTORS = {
    'CRYPTO': {'tickers': ['ETH-USD', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'LTC-USD'], 'benchmark': 'BTC-USD'},
    'MIDCAPS': {'tickers': ['ENPH', 'OKTA', 'CCJ', 'MDB', 'SHOP'], 'benchmark': 'IWM'},
    'COMMODITIES': {'tickers': ['GLD', 'SLV', 'USO', 'WEAT', 'CPER'], 'benchmark': 'DBC'}
}

def download_all_data():
    print("Telechargement des donnees (2016-2024)...")
    data = {}
    df_global = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_global.columns, pd.MultiIndex): df_global.columns = df_global.columns.droplevel(1)
    data['MASTER'] = df_global
    
    df_vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    data['VIX'] = df_vix['Close']
    
    for sector_name, info in SECTORS.items():
        print(f"  -> {sector_name}...")
        df_bench = yf.download(info['benchmark'], start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df_bench.columns, pd.MultiIndex): df_bench.columns = df_bench.columns.droplevel(1)
        sector_data = {'benchmark': df_bench['Close'], 'assets': {}}
        for ticker in info['tickers']:
            df_asset = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df_asset.columns, pd.MultiIndex): df_asset.columns = df_asset.columns.droplevel(1)
            if not df_asset.empty:
                sector_data['assets'][ticker] = df_asset
        data[sector_name] = sector_data
    return data

def prepare_features(df_asset, sector_name, s_bench, s_global, s_vix):
    df = df_asset.copy()
    df['Sector_Close'] = s_bench
    df['Global_Close'] = s_global
    if sector_name != 'CRYPTO' and not s_vix.empty:
        df['VIX'] = s_vix
        df['VIX'] = df['VIX'].ffill()
    
    df.dropna(subset=['Sector_Close', 'Global_Close'], inplace=True)
    
    df['Dist_SMA_10'] = (df['Close'] / df['Close'].rolling(10).mean()) - 1
    df['Dist_SMA_50'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
    df['Returns'] = df['Close'].pct_change().clip(lower=-0.15, upper=0.15)
    df['Returns_5d'] = df['Close'].pct_change(5).clip(lower=-0.25, upper=0.25)
    df['Returns_20d'] = df['Close'].pct_change(20).clip(lower=-0.40, upper=0.40)
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    df['Sector_Returns'] = df['Sector_Close'].pct_change()
    df['Sector_SMA_200'] = df['Sector_Close'].rolling(200).mean()
    df['Global_SMA_200'] = df['Global_Close'].rolling(200).mean()
    
    if 'VIX' in df.columns:
        df['VIX_Ratio'] = df['VIX'] / df['VIX'].rolling(30).mean()
    
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Relative_Strength_Sector'] = df['Close'].pct_change(10) - df['Sector_Close'].pct_change(10)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # V10 : Triple Barrier Labeling
    horizon = 15
    pt_mult = 2.0
    sl_mult = 1.0
    targets = np.zeros(len(df))
    close_prices = df['Close'].values
    vol_values = df['Volatility'].values
    for i in range(len(df) - horizon):
        if np.isnan(vol_values[i]) or vol_values[i] == 0: continue
        tp = close_prices[i] * (1 + (vol_values[i] * pt_mult * np.sqrt(horizon)))
        sl = close_prices[i] * (1 - (vol_values[i] * sl_mult * np.sqrt(horizon)))
        t = 0
        for j in range(i+1, i+horizon+1):
            if close_prices[j] >= tp: t = 1; break
            elif close_prices[j] <= sl: t = 0; break
        targets[i] = t
    df['Target'] = targets
    return df

class MLSignalData(bt.feeds.PandasData):
    """Custom Data Feed that includes ML Probability and Kelly Fraction"""
    lines = ('prob', 'kelly', 'regime_ok',)
    params = (
        ('prob', -1),
        ('kelly', -1),
        ('regime_ok', -1),
    )

class V10InstitutionalStrategy(bt.Strategy):
    params = (
        ('trailing_stop_pct', 0.05), # 5% trailing stop
        ('time_exit_days', 15),      # Sortie apres 15 jours si rien ne bouge
    )

    def __init__(self):
        self.entry_bars = {}
        self.stop_orders = {}

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                # On place le trailing stop une fois l'ordre d'achat execute
                stop_order = self.sell(data=order.data, exectype=bt.Order.StopTrail, trailpercent=self.params.trailing_stop_pct, size=order.executed.size)
                self.stop_orders[order.data._name] = stop_order
            elif order.issell():
                # L'ordre de vente (stop loss ou cloture) a ete execute
                if order.data._name in self.stop_orders:
                    self.stop_orders[order.data._name] = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

    def next(self):
        # 2. Rebalancement hebdomadaire (le lundi)
        if self.datetime.date(0).weekday() != 0:
            return

        # Calculer le budget total base sur le Kelly pour tous les signaux actifs
        candidates = []
        total_kelly = 0
        
        for data in self.datas:
            if len(data) > 0 and not pd.isna(data.prob[0]):
                prob = data.prob[0]
                kelly = data.kelly[0]
                regime = data.regime_ok[0]
                
                if regime and kelly > 0.05: # Minimum 5% de Kelly pour eviter le bruit
                    candidates.append((data, kelly))
                    total_kelly += kelly
                    
        # On ne garde que les 5 meilleurs selon le Kelly
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_5 = candidates[:5]
        top_5_kelly_sum = sum([c[1] for c in top_5])
        
        # Fermer les positions qui ne sont plus dans le top 5 ou qui ont atteint la limite de temps
        top_5_names = [c[0]._name for c in top_5]
        for data in self.datas:
            if self.getposition(data).size > 0:
                bars_held = len(self) - self.entry_bars.get(data._name, len(self))
                if data._name not in top_5_names or bars_held >= self.params.time_exit_days:
                    self.close(data=data)
                    # Annuler le trailing stop associe pour ne pas shorter accidentellement
                    if data._name in self.stop_orders and self.stop_orders[data._name] is not None:
                        self.cancel(self.stop_orders[data._name])
                        self.stop_orders[data._name] = None
                
        # Ouvrir / Ajuster les positions du Top 5 avec Trailing Stop
        for data, kelly in top_5:
            if top_5_kelly_sum > 0:
                target_pct = kelly / top_5_kelly_sum
                # Ne jamais investir plus de 95% pour garder du cash pour les frais
                target_pct = min(target_pct, 0.95 / len(top_5)) 
                
                if self.getposition(data).size == 0:
                    self.entry_bars[data._name] = len(self)
                    # Achat au marche (le trailing stop sera place dans notify_order)
                    self.order_target_percent(data=data, target=target_pct)
def run_backtest():
    print("--- PREPARATION DES DONNEES ET MODELES ML (V10) ---")
    data = download_all_data()
    df_global = data['MASTER']
    s_vix = data['VIX']
    
    # 1. Feature Engineering (Triple Barrier)
    features_dict = {}
    for sector_name, info in SECTORS.items():
        features_dict[sector_name] = {}
        feat_list = ['Dist_SMA_10', 'Dist_SMA_50', 'Returns', 'Returns_5d', 'Returns_20d',
                     'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector']
        if sector_name != 'CRYPTO': feat_list.extend(['VIX', 'VIX_Ratio'])
        
        for ticker, df_raw in data[sector_name]['assets'].items():
            df_feat = prepare_features(df_raw, sector_name, data[sector_name]['benchmark'], df_global['Close'], s_vix)
            features_dict[sector_name][ticker] = {'df': df_feat, 'features': feat_list}
            
    # 2. Entrainement Walk-Forward pour generer les predictions historiques
    print("Génération des prédictions ML (Walk-Forward)...")
    trading_days = df_global[df_global.index >= START_TRADING_DATE].index
    predictions_log = {}
    last_train_month = -1
    trained_models = {}
    
    # Pre-calcul des probabilités pour accélérer Backtrader
    for i, current_date in enumerate(trading_days):
        if current_date.month in [1, 4, 7, 10] and current_date.month != last_train_month:
            last_train_month = current_date.month
            for sector_name in SECTORS.keys():
                X_train_list, y_train_list = [], []
                for ticker, d in features_dict[sector_name].items():
                    df = d['df']
                    df_hist = df[(df.index < current_date)].dropna()
                    if len(df_hist) > 100 + 15:
                        # FIX LOOK-AHEAD BIAS: On retire les 15 derniers jours de l'historique
                        # car leur Target utilise des prix du futur qui n'ont pas encore eu lieu a current_date.
                        df_hist = df_hist.iloc[:-15]
                        X_train_list.append(df_hist[d['features']])
                        y_train_list.append(df_hist['Target'])
                if X_train_list:
                    X_train = pd.concat(X_train_list)
                    y_train = pd.concat(y_train_list)
                    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, n_jobs=-1, random_state=42)
                    model.fit(X_train, y_train)
                    trained_models[sector_name] = model
                    
        # Prediction chaque jour
        for sector_name, info in SECTORS.items():
            if sector_name not in trained_models: continue
            model = trained_models[sector_name]
            for ticker, d in features_dict[sector_name].items():
                if ticker not in predictions_log: predictions_log[ticker] = {}
                df = d['df']
                if current_date in df.index:
                    row = df.loc[current_date]
                    if not row[d['features']].isnull().any():
                        X_today = pd.DataFrame([row[d['features']]])
                        prob = model.predict_proba(X_today)[:, 1][0]
                        kelly = prob - ((1 - prob) / 2.0)
                        
                        reg_sec = row['Sector_Close'] > row['Sector_SMA_200']
                        reg_glob = row['Global_Close'] > row['Global_SMA_200']
                        reg_ok = reg_sec if sector_name == 'COMMODITIES' else (reg_glob and reg_sec)
                        
                        predictions_log[ticker][current_date] = {'prob': prob, 'kelly': kelly, 'regime_ok': int(reg_ok)}

    # 3. Execution Backtrader
    print("\n--- LANCEMENT BACKTRADER (V10 INSTITUTIONAL) ---")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=0.001) # 0.1% frais par trade
    
    # Ajouter les Data Feeds
    for sector_name, info in SECTORS.items():
        for ticker in info['tickers']:
            if ticker in features_dict[sector_name] and ticker in predictions_log:
                df = features_dict[sector_name][ticker]['df'].copy()
                df = df[df.index >= START_TRADING_DATE]
                
                # Ajouter les colonnes ML
                df['prob'] = np.nan
                df['kelly'] = np.nan
                df['regime_ok'] = 0
                for date, preds in predictions_log[ticker].items():
                    if date in df.index:
                        df.at[date, 'prob'] = preds['prob']
                        df.at[date, 'kelly'] = preds['kelly']
                        df.at[date, 'regime_ok'] = preds['regime_ok']
                        
                df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'prob', 'kelly', 'regime_ok']]
                data_feed = MLSignalData(dataname=df, name=ticker)
                cerebro.adddata(data_feed)

    cerebro.addstrategy(V10InstitutionalStrategy)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"Capital de depart : ${INITIAL_CAPITAL:,.2f}")
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f"Capital final     : ${final_value:,.2f}")
    
    strat = results[0]
    rendement = strat.analyzers.returns.get_analysis()['rtot'] * 100
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    trades = strat.analyzers.trades.get_analysis()
    
    print(f"Rendement total   : {rendement:+.2f}%")
    print(f"Max Drawdown      : -{max_dd:.2f}%")
    if 'total' in trades and 'closed' in trades['total']:
        print(f"Total trades      : {trades['total']['closed']}")
        print(f"Win Rate          : {trades['won']['total'] / trades['total']['closed'] * 100:.1f}%")

if __name__ == '__main__':
    run_backtest()
