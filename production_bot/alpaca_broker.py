import alpaca_trade_api as tradeapi
import config
import math
import time

def get_alpaca_api():
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise ValueError("Clés API Alpaca manquantes.")
    return tradeapi.REST(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_BASE_URL, api_version='v2')

def get_account_info(api):
    account = api.get_account()
    print(f"💰 Valeur du portefeuille : ${account.portfolio_value}")
    print(f"💵 Cash disponible : ${account.cash}")
    return account

def rebalance_portfolio(api, top_tickers):
    """
    1. Vend toutes les positions qui ne sont plus dans le Top N.
    2. Calcule le budget alloué pour chaque action du Top N.
    3. Achète/Ajuste les positions du Top N.
    """
    print("\n🔄 --- DÉBUT DU REBALANCEMENT ---")
    positions = api.list_positions()
    current_tickers = [p.symbol for p in positions]
    
    # 1. Vendre ce qui n'est plus dans le Top N
    for position in positions:
        if position.symbol not in top_tickers:
            print(f"🛑 Vente de la position {position.symbol} (N'est plus dans le Top {config.TOP_N})")
            api.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            time.sleep(1) # Eviter les limites d'API
            
    # Laisser le temps aux ordres de se remplir
    time.sleep(5)
    
    # 2. Calculer le budget par action
    account = api.get_account()
    total_equity = float(account.portfolio_value)
    # On garde 5% de cash par sécurité pour éviter les rejets pour manque de fonds
    safe_equity = total_equity * 0.95 
    budget_per_stock = safe_equity / len(top_tickers)
    
    print(f"🎯 Budget cible par action : ${budget_per_stock:.2f}")
    
    # 3. Acheter les actions du Top N
    for ticker in top_tickers:
        # Obtenir le prix actuel
        try:
            latest_trade = api.get_latest_trade(ticker)
            current_price = latest_trade.price
        except Exception as e:
            print(f"❌ Impossible d'obtenir le prix de {ticker}: {e}")
            continue
            
        target_qty = math.floor(budget_per_stock / current_price)
        
        # Obtenir la quantité actuelle
        current_qty = 0
        for p in api.list_positions():
            if p.symbol == ticker:
                current_qty = int(p.qty)
                break
                
        delta_qty = target_qty - current_qty
        
        if delta_qty > 0:
            print(f"🟢 Achat de {delta_qty} actions de {ticker}")
            api.submit_order(
                symbol=ticker,
                qty=delta_qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
        elif delta_qty < 0:
            print(f"🔴 Vente partielle de {abs(delta_qty)} actions de {ticker} (Rééquilibrage)")
            api.submit_order(
                symbol=ticker,
                qty=abs(delta_qty),
                side='sell',
                type='market',
                time_in_force='day'
            )
        else:
            print(f"✅ La position sur {ticker} est déjà parfaite.")
            
        time.sleep(1)
        
    print("✅ --- REBALANCEMENT TERMINÉ ---")
