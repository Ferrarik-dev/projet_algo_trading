import yfinance as yf
import pandas as pd
import numpy as np
import backtrader as bt
import lightgbm as lgb
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

def download_data_with_fundamentals():
    print("Téléchargement des prix et des fondamentaux...")
    df_prices = yf.download(UNIVERSE, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
    
    data = {}
    for ticker in UNIVERSE:
        if ticker in df_prices.columns.levels[0]:
            df_ticker = df_prices[ticker].copy()
            df_ticker.dropna(inplace=True)
            
            if not df_ticker.empty and len(df_ticker) > 252:
                # 1. Extraction des fondamentaux via yfinance
                ticker_obj = yf.Ticker(ticker)
                try:
                    df_fin = ticker_obj.quarterly_income_stmt.T
                    if not df_fin.empty and 'Net Income' in df_fin.columns:
                        df_fin.index = pd.to_datetime(df_fin.index)
                        df_fin = df_fin.sort_index()
                        
                        # SHIFT 45 JOURS (Lookahead Bias Protection)
                        df_fin.index = df_fin.index + pd.Timedelta(days=45)
                        
                        # Calcul de la croissance trimestrielle du bénéfice
                        df_fin['Net_Income_Growth'] = df_fin['Net Income'].pct_change()
                        
                        # Joindre avec les prix quotidiens (Forward Fill)
                        df_fin = df_fin[['Net Income', 'Net_Income_Growth']]
                        # Convert to timezone unaware to match prices
                        df_fin.index = df_fin.index.tz_localize(None) 
                        df_ticker.index = df_ticker.index.tz_localize(None)
                        
                        df_ticker = df_ticker.join(df_fin, how='left')
                        df_ticker['Net_Income_Growth'].ffill(inplace=True)
                        df_ticker['Net_Income_Growth'].fillna(0, inplace=True) # Fill missing early history with 0
                    else:
                        df_ticker['Net_Income_Growth'] = 0.0
                except Exception as e:
                    print(f"Erreur fondamentale pour {ticker}: {e}")
                    df_ticker['Net_Income_Growth'] = 0.0
                    
                data[ticker] = df_ticker
    return data

def prepare_features(df_raw):
    df = df_raw.copy()
    for lag in [1, 5, 21]:
        df[f'Ret_{lag}d'] = df['Close'].pct_change(lag)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
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
        if len(self.datas) > 0:
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
        
        target_pct = 0.95 / self.params.top_n if top_candidates else 0
        
        top_datas = [c[0] for c in top_candidates]
        for data in self.datas:
            if data in top_datas:
                self.order_target_percent(data=data, target=target_pct)
            else:
                self.order_target_percent(data=data, target=0.0)

def run_simulation():
    data = download_data_with_fundamentals()
    all_features = {ticker: prepare_features(df) for ticker, df in data.items()}
    
    features_cols = ['Ret_1d', 'Ret_5d', 'Ret_21d', 'RSI_14', 'MACD', 'DayOfWeek', 'Month', 'Net_Income_Growth']
    
    dates_index = pd.date_range(start=START_TRADING_DATE, end=END_DATE, freq='QE')
    predictions_log = {ticker: {} for ticker in UNIVERSE}
    
    for i in range(len(dates_index) - 1):
        quarter_start = dates_index[i]
        quarter_end = dates_index[i+1]
        
        X_train, y_train = [], []
        for ticker, df in all_features.items():
            df_hist = df[df.index < quarter_start]
            if len(df_hist) > 252 + 5:
                df_hist = df_hist.iloc[:-5] 
                X_train.append(df_hist[features_cols])
                y_train.append(df_hist['Target_5d'])
                
        if len(X_train) == 0:
            print(f"Skipping quarter {quarter_start.date()} : X_train is empty")
            continue
            
        X_train_df = pd.concat(X_train)
        y_train_df = pd.concat(y_train)
        
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train_df, y_train_df)
        
        for ticker, df in all_features.items():
            df_test = df[(df.index >= quarter_start) & (df.index < quarter_end)]
            if not df_test.empty:
                preds = model.predict(df_test[features_cols])
                for date, p in zip(df_test.index, preds):
                    predictions_log[ticker][date] = p
                    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    
    print(f"Total entries in predictions_log for AAPL: {len(predictions_log.get('AAPL', {}))}")
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

    cerebro.addstrategy(StefanJansenRankingStrategy, top_n=3)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    roi = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    
    print(f"Capital final   : ${final_value:,.2f}")
    print(f"Rendement Absolu: +{roi:.2f}%")
    print(f"Max Drawdown    : -{max_dd:.2f}%")
    
    return strat

if __name__ == '__main__':
    print("DEMARRAGE DU SETUP STEFAN JANSEN + DONNEES FONDAMENTALES")
    import matplotlib.pyplot as plt
    strat = run_simulation()
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat.dates, strat.portfolio_values, color='cyan', linewidth=2)
    plt.title('Performance du Bot : 20 Mega-Caps + Fondamentaux', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur du Portefeuille ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle='-', linewidth=1)
    artifact_path = r'C:\Users\Elrik\.gemini\antigravity\brain\3bd380e5-0c43-4284-9645-00b7bc827801\mega_fundamentals_equity_curve.png'
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=300)
    print(f"Graphique sauvegardé : {artifact_path}")
