import pandas as pd

def prepare_features(df_raw):
    """
    Calcule exactement les mêmes indicateurs techniques que dans le backtest Baseline (+362%).
    """
    df = df_raw.copy()
    
    # Rendements passés
    for lag in [1, 5, 21, 42, 63, 126, 252]:
        df[f'Ret_{lag}d'] = df['Close'].pct_change(lag)
        
    # Volatilité
    df['Vol_21d'] = df['Ret_1d'].rolling(21).std()
    
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
    
    # Bandes de Bollinger
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['Dist_BB_Upper'] = (df['Close'] / df['BB_Upper']) - 1
    df['Dist_BB_Lower'] = (df['Close'] / df['BB_Lower']) - 1
    
    # Saisonnalité
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    # Target (Seulement pour l'entrainement, sera NaN pour aujourd'hui)
    df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # On ne fait PAS de dropna ici car on a besoin de la dernière ligne (aujourd'hui) même si Target_5d est NaN
    return df

def get_feature_list():
    return [
        'Ret_1d', 'Ret_5d', 'Ret_21d', 'Ret_42d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'Vol_21d', 'RSI_14', 'MACD', 'Dist_BB_Upper', 'Dist_BB_Lower', 'DayOfWeek', 'Month'
    ]
