import yfinance as yf
from transformers import pipeline
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement de l'Intelligence Artificielle "FinBERT" (Façon Stefan Jansen)
# C'est un modèle de langage (LLM) pré-entraîné sur des millions d'articles financiers.
print("Chargement du cerveau NLP FinBERT (Cela peut prendre quelques secondes)...")
nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 2. Liste d'actions à analyser (Un sous-ensemble de Mega-Caps pour le test)
TARGET_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

print("\n--- MOTEUR NLP DE LECTURE DES NEWS (LIVE) ---\n")

for ticker in TARGET_STOCKS:
    print(f"Extraction des gros titres pour {ticker}...")
    stock = yf.Ticker(ticker)
    news_items = stock.news
    
    if not news_items:
        print(f"  Aucune nouvelle trouvée pour {ticker}.")
        continue
        
    headlines = []
    for item in news_items[:5]:
        if 'content' in item and 'title' in item['content']:
            headlines.append(item['content']['title'])
        elif 'title' in item:
            headlines.append(item['title'])
            
    if not headlines:
        print(f"  Aucun titre lisible trouvé pour {ticker}.")
        continue
    
    total_score = 0
    print(f"Analyse FinBERT en cours pour {ticker}:")
    
    for headline in headlines:
        # L'IA lit le titre et donne son sentiment
        result = nlp(headline)[0]
        sentiment = result['label']
        confidence = result['score']
        
        # Conversion du sentiment en score mathématique
        if sentiment == 'positive':
            score = 1 * confidence
        elif sentiment == 'negative':
            score = -1 * confidence
        else:
            score = 0
            
        total_score += score
        
        # Affichage visuel
        if score > 0.1:
            emoji = "[+]"
        elif score < -0.1:
            emoji = "[-]"
        else:
            emoji = "[=]"
            
        print(f"  {emoji} [{sentiment.upper()}] (Confiance: {confidence:.2f}) -> {headline}")
        
    avg_score = total_score / len(headlines)
    
    # Verdict de l'action
    if avg_score > 0.2:
        verdict = "FEU VERT (Tendance haussière détectée par l'IA)"
    elif avg_score < -0.2:
        verdict = "FEU ROUGE (Danger, panique médiatique)"
    else:
        verdict = "NEUTRE (Bruit médiatique normal)"
        
    print(f"\n=> SCORE GLOBAL POUR {ticker} : {avg_score:.2f} | VERDICT : {verdict}\n")
    print("-" * 80)
