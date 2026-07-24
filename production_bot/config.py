import os

# --- UNIVERS D'ACTIONS ---
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'BAC', 'COST'
]

# --- PARAMETRES DU MODELE ---
TOP_N = 3
BEST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# --- PARAMETRES D'ENTRAINEMENT ---
# En production, on s'entraîne sur les 5 dernières années d'historique
YEARS_HISTORY = 5

# --- API KEYS ---
# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" # PAPER TRADING !

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
