import sys
from data_loader import download_historical_data
from feature_engine import prepare_features
from model_trainer import train_and_predict
from alpaca_broker import get_alpaca_api, get_account_info, rebalance_portfolio
from telegram_notifier import send_telegram_message
import config

def main():
    try:
        msg = "🚀 <b>Début de l'analyse quotidienne de l'algorithme !</b>"
        print(msg)
        send_telegram_message(msg)
        
        # 1. Téléchargement des données
        raw_data = download_historical_data()
        
        # 2. Feature Engineering
        all_features = {}
        for ticker, df in raw_data.items():
            all_features[ticker] = prepare_features(df)
            
        # 3. Prédictions
        df_preds = train_and_predict(all_features)
        
        print("\n🏆 CLASSEMENT DU JOUR :")
        print(df_preds.head(10))
        
        # 4. Sélection du Top N (Uniquement ceux avec un rendement positif)
        top_candidates = df_preds[df_preds['predicted_return'] > 0].head(config.TOP_N)
        top_tickers = top_candidates['ticker'].tolist()
        
        msg_top = f"📊 <b>Top {config.TOP_N} du jour sélectionné :</b>\n"
        for idx, row in top_candidates.iterrows():
            msg_top += f"{idx+1}. {row['ticker']} (+{row['predicted_return']:.2%})\n"
        
        print(f"\n{msg_top}")
        send_telegram_message(msg_top)
        
        # 5. Exécution Alpaca
        api = get_alpaca_api()
        account_before = get_account_info(api)
        
        rebalance_portfolio(api, top_tickers)
        
        account_after = get_account_info(api)
        msg_end = f"✅ <b>Rebalancement terminé avec succès.</b>\n💰 Valeur du portefeuille : ${account_after.portfolio_value}"
        print(msg_end)
        send_telegram_message(msg_end)
        
    except Exception as e:
        error_msg = f"❌ <b>ERREUR CRITIQUE DANS LE BOT :</b>\n{str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
