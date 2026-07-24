import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import config

def download_historical_data():
    """
    Télécharge les données pour l'entraînement et l'inférence.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.YEARS_HISTORY * 365)
    
    data = {}
    for ticker in config.UNIVERSE:
        # On télécharge les données, progress=False pour éviter de polluer les logs GitHub Actions
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        if not df.empty:
            data[ticker] = df
            
    return data
