import backtrader as bt
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import numpy as np
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# === 1. PARAMETRES DU SETUP STEFAN JANSEN (Mega-Caps) ===
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 
            'PG', 'UNH', 'HD', 'DIS', 'MA', 'PYPL', 'VZ', 'ADBE', 'NFLX', 'INTC']

START_DATE = '2016-01-01'
END_DATE = '2023-12-31'
START_TRADING_DATE = '2019-01-01'
INITIAL_CAPITAL = 10000.0

# === 2. GESTION DES DONNEES ET FEATURE ENGINEERING ===
def get_data():
    if os.path.exists('data_mega_caps.csv'):
        df = pd.read_csv('data_mega_caps.csv', index_col=[0, 1], parse_dates=True)
    else:
        print("T?l?chargement des donn?es...")
        data = yf.download(UNIVERSE, start=START_DATE, end=END_DATE, group_by='ticker')
        df_list = []
        for ticker in UNIVERSE:
            if ticker in data.columns.levels[0]:
                df_ticker = data[ticker].copy()
                df_ticker['ticker'] = ticker
                df_list.append(df_ticker)
        
        df = pd.concat(df_list)
        df.reset_index(inplace=True)
        df.set_index(['Date', 'ticker'], inplace=True)
        df.to_csv('data_mega_caps.csv')
    return df

def prepare_features(df):
    df['Ret_1d'] = df['Close'].pct_change()
    df['Ret_5d'] = df['Close'].pct_change(5)
    df['Ret_21d'] = df['Close'].pct_change(21)
    df['Vol_21d'] = df['Ret_1d'].rolling(21).std()
    
    # Target: rendement futur ? 5 jours
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # SMA 200 Trend Filter
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Uptrend'] = (df['Close'] > df['SMA_200']).astype(int)
    
    df.dropna(inplace=True)
    return df

# === 3. MACHINE LEARNING (LIGHTGBM) ===
def train_and_predict(all_features):
    predictions_log = {}
    features_cols = ['Ret_1d', 'Ret_5d', 'Ret_21d', 'Vol_21d']
    
    # Quarterly walk-forward
    start_test = pd.to_datetime(START_TRADING_DATE)
    end_test = pd.to_datetime(END_DATE)
    quarters = pd.date_range(start_test, end_test, freq='QS')
    
    model = lgb.LGBMRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42, 
        verbose=-1
    )
    
    for quarter_start in quarters:
        X_train, y_train = [], []
        
        for ticker, df in all_features.items():
            df_hist = df[df.index < quarter_start]
            if len(df_hist) > 252:
                # Remove last 5 days to avoid target leak
                df_hist = df_hist.iloc[:-5] 
                X_train.append(df_hist[features_cols])
                y_train.append(df_hist['Target_5d'])
                
        if len(X_train) == 0:
            continue
            
        X_train_df = pd.concat(X_train)
        y_train_df = pd.concat(y_train)
        
        model.fit(X_train_df, y_train_df)
        
        quarter_end = quarter_start + pd.DateOffset(months=3)
        for ticker, df in all_features.items():
            df_test = df[(df.index >= quarter_start) & (df.index < quarter_end)]
            if len(df_test) > 0:
                preds = model.predict(df_test[features_cols])
                if ticker not in predictions_log:
                    predictions_log[ticker] = pd.Series(dtype=float)
                
                # Appliquer le filtre SMA 200: si pas Uptrend, on met la pr?diction ? une valeur n?gative extr?me
                preds_filtered = np.where(df_test['Uptrend'] == 1, preds, -999)
                predictions_log[ticker] = pd.concat([predictions_log[ticker], pd.Series(preds_filtered, index=df_test.index)])
                
    return predictions_log

# === 4. STRATEGIE BACKTRADER (RISK PARITY) ===
class MLRiskParityStrategy(bt.Strategy):
    params = (
        ('top_n', 3),
    )

    def __init__(self):
        self.predictions = self.broker.cerebro.predictions
        self.rebalance_days = 0

    def next(self):
        current_date = self.data.datetime.date(0)
        
        # R?balancement tous les 5 jours (Hebdomadaire)
        if self.rebalance_days % 5 != 0:
            self.rebalance_days += 1
            return
            
        self.rebalance_days += 1
        
        preds_today = {}
        for d in self.datas:
            ticker = d._name
            if ticker in self.predictions:
                pred_series = self.predictions[ticker]
                # Convert backtrader date to pandas timestamp for lookup
                pd_date = pd.Timestamp(current_date)
                if pd_date in pred_series.index:
                    pred_val = pred_series.loc[pd_date]
                    # Only consider positive predictions (and those passing SMA200 filter)
                    if pred_val > 0:
                        preds_today[ticker] = pred_val

        # Sort and select Top N
        sorted_preds = sorted(preds_today.items(), key=lambda x: x[1], reverse=True)
        selected_tickers = [x[0] for x in sorted_preds[:self.p.top_n]]
        
        # --- LOGIQUE RISK PARITY ---
        inv_vols = {}
        for d in self.datas:
            if d._name in selected_tickers:
                # Obtenir l'historique des prix sur 21 jours pour calculer la volatilit?
                closes = np.array(d.close.get(size=21))
                if len(closes) < 21:
                    continue # Pas assez d'historique, on ignore
                
                returns = np.diff(closes) / closes[:-1]
                vol = np.std(returns)
                
                # Pr?vention division par z?ro
                if vol == 0 or np.isnan(vol):
                    vol = 1e-6
                    
                # La pond?ration est l'inverse du risque (volatilit?)
                inv_vols[d._name] = 1.0 / vol
                
        # Calculer le poids final de chaque action (Normalisation pour que la somme = 100%)
        total_inv_vol = sum(inv_vols.values())
        weights = {}
        if total_inv_vol > 0:
            for ticker, iv in inv_vols.items():
                weights[ticker] = iv / total_inv_vol
        
        # --- PASSAGE DES ORDRES ---
        for d in self.datas:
            if d._name in weights:
                # Allocation proportionnelle au Risk Parity
                self.order_target_percent(d, target=weights[d._name])
            else:
                self.order_target_percent(d, target=0.0)

# === 5. SETUP BACKTRADER ===
def run_simulation():
    df_raw = get_data()
    
    all_features = {}
    for ticker in UNIVERSE:
        if ticker in df_raw.index.get_level_values('ticker'):
            df_t = df_raw.xs(ticker, level='ticker').copy()
            df_t = prepare_features(df_t)
            all_features[ticker] = df_t
            
    print("Entrainement LightGBM et Generation des predictions...")
    predictions_log = train_and_predict(all_features)
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    
    for ticker, df in all_features.items():
        if ticker in predictions_log:
            df_bt = df[df.index >= START_TRADING_DATE].copy()
            if len(df_bt) > 0:
                data = bt.feeds.PandasData(dataname=df_bt, name=ticker)
                cerebro.adddata(data)
                
    cerebro.predictions = predictions_log
    cerebro.addstrategy(MLRiskParityStrategy, top_n=3)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print("Demarrage du Backtest (Risk Parity)...")
    results = cerebro.run()
    
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    
    print(f"Capital final   : ${final_value:,.2f}")
    roi = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"Rendement Absolu: +{roi:.2f}%")
    print(f"Max Drawdown    : -{drawdown:.2f}%")
    
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + RISK PARITY")
    import matplotlib.pyplot as plt
    strat = run_simulation()
    
    plt.figure(figsize=(12,6))
    plt.plot(strat.observers.broker[0].lines.value.array, color='blue')
    plt.title('Equity Curve - Machine Learning + Risk Parity')
    plt.grid(True)
    plt.savefig('C:\\Users\\Elrik\\.gemini\\antigravity\\brain\\3bd380e5-0c43-4284-9645-00b7bc827801\\risk_parity_equity_curve.png')
    print("Graphique sauvegarde : risk_parity_equity_curve.png")
