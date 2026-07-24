import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

START_DATE = '2016-01-01'
END_DATE = '2024-01-01'
START_TRADING_DATE = '2017-01-01'
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

def download_all_data():
    print("Telechargement des donnees (2016-2024)...")
    data = {}
    df_global = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_global.columns, pd.MultiIndex): df_global.columns = df_global.columns.droplevel(1)
    data['MASTER'] = df_global
    
    df_vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.droplevel(1)
    data['VIX'] = df_vix['Close']
    
    print("  -> DXY + TNX...")
    df_dxy = yf.download('DX-Y.NYB', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_dxy.columns, pd.MultiIndex): df_dxy.columns = df_dxy.columns.droplevel(1)
    data['DXY'] = df_dxy['Close'] if not df_dxy.empty else pd.Series(dtype=float)
    
    df_tnx = yf.download('^TNX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df_tnx.columns, pd.MultiIndex): df_tnx.columns = df_tnx.columns.droplevel(1)
    data['TNX'] = df_tnx['Close'] if not df_tnx.empty else pd.Series(dtype=float)
    
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
    
    # Regime Detection : Volatilite relative
    df['Vol_20d'] = df['Returns'].rolling(20).std()
    df['Vol_60d_Avg'] = df['Vol_20d'].rolling(60).mean()
    # 0=Calme, 1=Normal, 2=Crise
    df['Vol_Regime'] = np.where(df['Vol_20d'] > df['Vol_60d_Avg'] * 1.5, 2,
                       np.where(df['Vol_20d'] < df['Vol_60d_Avg'] * 0.8, 0, 1))
    
    df['Target'] = (df['Close'].shift(-15) > df['Close']).astype(int)
    return df

def detect_macro_stress_at_date(dxy_series, tnx_series, vix_series, current_date):
    """Score de stress macro a une date donnee (0=OK, 1-3=Danger)."""
    stress = 0
    
    dxy_before = dxy_series[dxy_series.index <= current_date]
    if len(dxy_before) > 20:
        dxy_chg = (dxy_before.iloc[-1] / dxy_before.iloc[-20]) - 1
        if dxy_chg > 0.03:
            stress += 1
    
    tnx_before = tnx_series[tnx_series.index <= current_date]
    if len(tnx_before) > 20:
        tnx_chg = tnx_before.iloc[-1] - tnx_before.iloc[-20]
        if tnx_chg > 0.30:
            stress += 1
    
    vix_before = vix_series[vix_series.index <= current_date]
    if len(vix_before) > 0 and vix_before.iloc[-1] > 30:
        stress += 1
    
    return stress

def run_simulation(data, mode='v8'):
    labels = {
        'v8': 'V8 PRO (Base)',
        'v9_hybrid': 'V9 Hybrid (Filtre Macro + Regime)'
    }
    label = labels[mode]
    print(f"\n--- Simulation {label} ---")
    
    df_global = data['MASTER']
    s_vix = data['VIX']
    s_dxy = data.get('DXY', pd.Series(dtype=float))
    s_tnx = data.get('TNX', pd.Series(dtype=float))
    
    features_dict = {}
    for sector_name, info in SECTORS.items():
        features_dict[sector_name] = {}
        feat_list = ['Dist_SMA_10', 'Dist_SMA_50', 'Returns', 'Returns_5d', 'Returns_20d',
                     'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector']
        if sector_name != 'CRYPTO':
            feat_list.extend(['VIX', 'VIX_Ratio'])
        
        for ticker, df_raw in data[sector_name]['assets'].items():
            df_feat = prepare_features(df_raw, sector_name, data[sector_name]['benchmark'],
                                       df_global['Close'], s_vix)
            features_dict[sector_name][ticker] = {'df': df_feat, 'features': feat_list}
    
    trading_days = df_global[df_global.index >= START_TRADING_DATE].index
    capital = INITIAL_CAPITAL
    equity_curve = []
    current_portfolio = {}
    trained_models = {}
    last_train_month = -1
    macro_blocks = 0  # Compteur de fois ou le filtre macro a bloque
    regime_blocks = 0
    
    for i, current_date in enumerate(trading_days):
        # 1. Rendement quotidien
        if i > 0 and len(current_portfolio) > 0:
            daily_return = 0
            for ticker, weight in current_portfolio.items():
                sec = None
                for s, info in SECTORS.items():
                    if ticker in info['tickers']: sec = s
                if sec and ticker in features_dict[sec] and current_date in features_dict[sec][ticker]['df'].index:
                    ret = features_dict[sec][ticker]['df'].at[current_date, 'Returns']
                    if not pd.isna(ret):
                        daily_return += ret * weight
            capital *= (1 + daily_return)
        equity_curve.append({'Date': current_date, 'Equity': capital})
        
        # 2. Entrainement trimestriel
        if current_date.month in [1, 4, 7, 10] and current_date.month != last_train_month:
            last_train_month = current_date.month
            for sector_name in SECTORS.keys():
                X_train_list, y_train_list = [], []
                for ticker, d in features_dict[sector_name].items():
                    df = d['df']
                    df_hist = df[(df.index < current_date)].dropna()
                    if len(df_hist) > 100:
                        X_train_list.append(df_hist[d['features']])
                        y_train_list.append(df_hist['Target'])
                if X_train_list:
                    X_train = pd.concat(X_train_list)
                    y_train = pd.concat(y_train_list)
                    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, n_jobs=-1, random_state=42)
                    model.fit(X_train, y_train)
                    trained_models[sector_name] = model
        
        # 3. Rebalancement hebdomadaire (lundi)
        if current_date.weekday() == 0:
            # V9 HYBRID : Filtre de Protection Macro
            if mode == 'v9_hybrid':
                macro_stress = detect_macro_stress_at_date(s_dxy, s_tnx, s_vix, current_date)
                if macro_stress >= 2:
                    # FREIN D'URGENCE : on liquide tout, on passe en cash
                    current_portfolio = {}
                    macro_blocks += 1
                    continue
            
            todays_predictions = []
            for sector_name, info in SECTORS.items():
                if sector_name not in trained_models: continue
                model = trained_models[sector_name]
                for ticker, d in features_dict[sector_name].items():
                    df = d['df']
                    if current_date in df.index:
                        row = df.loc[current_date]
                        regime_sector_ok = row['Sector_Close'] > row['Sector_SMA_200']
                        regime_global_ok = row['Global_Close'] > row['Global_SMA_200']
                        regime_ok = regime_sector_ok if sector_name == 'COMMODITIES' else (regime_global_ok and regime_sector_ok)
                        
                        # V9 HYBRID : Regime de Volatilite
                        if mode == 'v9_hybrid' and 'Vol_Regime' in row.index:
                            if row['Vol_Regime'] == 2:  # Crise : on n'achete pas ce ticker
                                regime_blocks += 1
                                continue
                        
                        if regime_ok and not row[d['features']].isnull().any():
                            X_today = pd.DataFrame([row[d['features']]])
                            prob = model.predict_proba(X_today)[:, 1][0]
                            if prob >= 0.60:
                                todays_predictions.append({'ticker': ticker, 'prob': prob})
            
            todays_predictions.sort(key=lambda x: x['prob'], reverse=True)
            top_5 = [p['ticker'] for p in todays_predictions[:5]]
            new_portfolio = {}
            if len(top_5) > 0:
                weight = 1.0 / len(top_5)
                for t in top_5:
                    new_portfolio[t] = weight
            current_portfolio = new_portfolio
    
    df_equity = pd.DataFrame(equity_curve)
    algo_ret = (capital / INITIAL_CAPITAL) - 1
    df_equity['Peak'] = df_equity['Equity'].cummax()
    df_equity['Drawdown'] = (df_equity['Peak'] - df_equity['Equity']) / df_equity['Peak']
    max_dd = df_equity['Drawdown'].max()
    
    print(f"  Capital Final    : ${capital:,.2f}")
    print(f"  Rendement        : {algo_ret*100:+.2f}%")
    print(f"  Max Drawdown     : -{max_dd*100:.2f}%")
    if mode == 'v9_hybrid':
        print(f"  Macro Blocks     : {macro_blocks} semaines en cash (protection)")
        print(f"  Regime Blocks    : {regime_blocks} signaux bloques (vol trop haute)")
    
    return df_equity, capital, algo_ret, max_dd

if __name__ == '__main__':
    data = download_all_data()
    
    eq_v8, cap_v8, ret_v8, dd_v8 = run_simulation(data, mode='v8')
    eq_v9h, cap_v9h, ret_v9h, dd_v9h = run_simulation(data, mode='v9_hybrid')
    
    # SPY
    df_spy = data['MASTER']
    df_spy = df_spy[(df_spy.index >= START_TRADING_DATE) & (df_spy.index <= END_DATE)]
    spy_normalized = (df_spy['Close'] / df_spy['Close'].iloc[0]) * INITIAL_CAPITAL
    spy_ret = (df_spy['Close'].iloc[-1] / df_spy['Close'].iloc[0]) - 1
    
    print("\n" + "="*55)
    print("  COMPARAISON V8 vs V9 HYBRID (Filtre+Regime) - 8 ANS")
    print("="*55)
    print(f"  S&P 500          : {spy_ret*100:+.2f}%")
    print(f"  V8 PRO           : {ret_v8*100:+.2f}% | Drawdown: -{dd_v8*100:.2f}%")
    print(f"  V9 HYBRID        : {ret_v9h*100:+.2f}% | Drawdown: -{dd_v9h*100:.2f}%")
    print(f"  Delta Rendement  : {(ret_v9h - ret_v8)*100:+.2f}%")
    print(f"  Delta Drawdown   : {(dd_v8 - dd_v9h)*100:+.2f}% (positif = meilleur)")
    print("="*55)
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(spy_normalized.index, spy_normalized, label=f'SPY ({spy_ret*100:+.1f}%)', color='grey', linewidth=2)
    eq_v8.set_index('Date', inplace=True)
    ax1.plot(eq_v8.index, eq_v8['Equity'], label=f'V8 PRO ({ret_v8*100:+.1f}%)', color='orange', linewidth=2, linestyle='--', alpha=0.7)
    eq_v9h.set_index('Date', inplace=True)
    ax1.plot(eq_v9h.index, eq_v9h['Equity'], label=f'V9 HYBRID ({ret_v9h*100:+.1f}%)', color='#00FF88', linewidth=2.5)
    ax1.set_title('V8 vs V9 HYBRID (Filtre Macro + Detection de Regime) - 8 Ans', fontsize=14)
    ax1.set_ylabel('Capital ($)')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Drawdown
    ax2.fill_between(eq_v8.index, -eq_v8['Drawdown']*100, 0, alpha=0.3, color='orange', label=f'V8 DD (max -{dd_v8*100:.1f}%)')
    ax2.fill_between(eq_v9h.index, -eq_v9h['Drawdown']*100, 0, alpha=0.5, color='#00FF88', label=f'V9H DD (max -{dd_v9h*100:.1f}%)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_v9_hybrid.png', bbox_inches='tight', dpi=120)
    print("Graphique sauvegarde sous 'backtest_v9_hybrid.png'")
