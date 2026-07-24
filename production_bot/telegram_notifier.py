import requests
import config

def send_telegram_message(message):
    """
    Envoie un message via Telegram en utilisant le bot.
    """
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("⚠️ Credentials Telegram manquants. Message non envoyé.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Message Telegram envoyé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message Telegram: {e}")
