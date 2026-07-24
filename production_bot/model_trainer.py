import pandas as pd
import lightgbm as lgb
import config
from feature_engine import get_feature_list

def train_and_predict(all_features_dict):
    """
    Entraîne le modèle sur tout l'historique disponible et prédit le classement du jour.
    """
    features_list = get_feature_list()
    
    X_train_list, y_train_list = [], []
    X_today_dict = {}
    
    print("⏳ Préparation des données d'entraînement...")
    for ticker, df in all_features_dict.items():
        # Pour l'entrainement, on exclut les 5 derniers jours car Target_5d est NaN
        df_train = df.dropna(subset=['Target_5d', *features_list])
        
        if not df_train.empty:
            X_train_list.append(df_train[features_list])
            y_train_list.append(df_train['Target_5d'])
            
        # Pour la prédiction d'aujourd'hui, on prend la toute dernière ligne
        last_row = df.iloc[[-1]]
        # On vérifie qu'on a bien toutes les features (ex: pas de NaN sur RSI parce qu'il manque d'historique)
        if not last_row[features_list].isnull().any().any():
            X_today_dict[ticker] = last_row[features_list]
            
    if not X_train_list:
        raise ValueError("Aucune donnée d'entraînement disponible.")
        
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    print(f"🤖 Entraînement du modèle LightGBM sur {len(X_train)} lignes d'historique...")
    model = lgb.LGBMRegressor(**config.BEST_PARAMS)
    model.fit(X_train, y_train)
    
    print("🔮 Génération des prédictions pour aujourd'hui...")
    predictions = []
    for ticker, x_today in X_today_dict.items():
        pred = model.predict(x_today)[0]
        predictions.append({'ticker': ticker, 'predicted_return': pred})
        
    df_preds = pd.DataFrame(predictions)
    # Trier par prédiction décroissante
    df_preds = df_preds.sort_values(by='predicted_return', ascending=False).reset_index(drop=True)
    
    return df_preds
