import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    print("Téléchargement des données de test (AAPL, MSFT, QQQ, ^VIX)...")
    data = {}
    data['AAPL'] = yf.download('AAPL', start='2015-01-01', end='2026-01-01', progress=False)
    data['MSFT'] = yf.download('MSFT', start='2015-01-01', end='2026-01-01', progress=False)
    data['QQQ'] = yf.download('QQQ', start='2015-01-01', end='2026-01-01', progress=False)
    data['VIX'] = yf.download('^VIX', start='2015-01-01', end='2026-01-01', progress=False)
    
    # Fix multi-index
    for k, df in data.items():
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
    return data

def prepare_features(df_asset, df_bench, df_vix, mode='new'):
    df = pd.DataFrame()
    df['Close'] = df_asset['Close']
    df['Volume'] = df_asset['Volume']
    df['Sector_Close'] = df_bench['Close']
    df['VIX'] = df_vix['Close']
    df.dropna(inplace=True)
    
    # Common features
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Sector_Returns'] = df['Sector_Close'].pct_change()
    df['Relative_Strength_Sector'] = df['Close'].pct_change(10) - df['Sector_Close'].pct_change(10)
    df['VIX_Ratio'] = df['VIX'] / df['VIX'].rolling(30).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    if mode == 'old':
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(20).std()
        features = ['SMA_10', 'SMA_50', 'Returns', 'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector', 'VIX', 'VIX_Ratio']
    else:
        df['Dist_SMA_10'] = (df['Close'] / df['Close'].rolling(10).mean()) - 1
        df['Dist_SMA_50'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        df['Returns'] = df['Close'].pct_change().clip(lower=-0.15, upper=0.15)
        df['Returns_5d'] = df['Close'].pct_change(5).clip(lower=-0.25, upper=0.25)
        df['Returns_20d'] = df['Close'].pct_change(20).clip(lower=-0.40, upper=0.40)
        df['Volatility'] = df['Returns'].rolling(20).std()
        features = ['Dist_SMA_10', 'Dist_SMA_50', 'Returns', 'Returns_5d', 'Returns_20d', 'Volatility', 'Volume_Ratio', 'RSI_14', 'MACD', 'Sector_Returns', 'Relative_Strength_Sector', 'VIX', 'VIX_Ratio']
        
    df['Target'] = (df['Close'].shift(-15) > df['Close']).astype(int)
    
    df = df.dropna()
    return df, features

def evaluate_model(data, mode):
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    for ticker in ['AAPL', 'MSFT']:
        df, features = prepare_features(data[ticker], data['QQQ'], data['VIX'], mode)
        X = df[features]
        y = df['Target']
        
        # Walk-Forward Validation (5 splits)
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
            
    acc = accuracy_score(all_y_true, all_y_pred)
    loss = log_loss(all_y_true, all_y_prob)
    return acc, loss

if __name__ == '__main__':
    print("=== DEBUT DE LA SIMULATION ===")
    data = fetch_data()
    
    print("\n[1/2] Evaluation de l'ANCIEN modèle (Prix bruts, pas de momentum)...")
    acc_old, loss_old = evaluate_model(data, mode='old')
    print(f"-> Accuracy : {acc_old*100:.2f}%")
    print(f"-> Log-Loss : {loss_old:.4f}")
    
    print("\n[2/2] Evaluation du NOUVEAU modèle (Stationnaire, Multi-Timeframe, Clip)...")
    acc_new, loss_new = evaluate_model(data, mode='new')
    print(f"-> Accuracy : {acc_new*100:.2f}%")
    print(f"-> Log-Loss : {loss_new:.4f}")
    
    print("\n=== CONCLUSION ===")
    print(f"Amélioration de l'Accuracy : {(acc_new - acc_old)*100:+.2f} points de pourcentage")
    print(f"Amélioration de l'Erreur (Log-Loss) : {(loss_new - loss_old):+.4f} (plus bas = meilleur)")
    
    if acc_new > acc_old and loss_new < loss_old:
        print("\n✅ SUCCÈS : Le nouveau Feature Engineering est mathématiquement supérieur.")
    else:
        print("\n⚠️ AVERTISSEMENT : Les résultats sont mitigés.")
