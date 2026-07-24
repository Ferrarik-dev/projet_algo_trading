import yfinance as yf
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_telegram_notification(message):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("Telegram non configuré.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erreur Telegram: {e}")

def check_alerts():
    assets = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq', 'BTC-USD': 'Bitcoin'}
    alerts = []
    
    for ticker, name in assets.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if len(data) >= 2:
                # Extraction sécurisée des prix (gestion des index multi-niveaux de yfinance)
                close_series = data['Close'] if 'Close' in data else data.iloc[:, data.columns.get_level_values(0)=='Close']
                
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                    
                prev_close = float(close_series.iloc[-2])
                curr_price = float(close_series.iloc[-1])
                
                pct_change = ((curr_price - prev_close) / prev_close) * 100
                
                # Seuil de 3%
                if abs(pct_change) >= 3.0:
                    direction = "🚀 POMPE" if pct_change > 0 else "🩸 KRACH"
                    alerts.append(f"<b>{direction} DETECTE : {name} ({ticker})</b>\nVariation : {pct_change:+.2f}%\nPrix Actuel : {curr_price:.2f} $")
        except Exception as e:
            print(f"Erreur avec {ticker}: {e}")
            
    if alerts:
        message = "🚨 <b>ALERTE MARCHE V8 PRO</b> 🚨\n\n" + "\n\n".join(alerts)
        send_telegram_notification(message)
        print("Alerte envoyée:\n", message)
    else:
        print("Marché stable. Pas d'alerte.")

if __name__ == "__main__":
    import pandas as pd # required for the isinstance check
    print("Démarrage du scanner d'alertes live...")
    check_alerts()
