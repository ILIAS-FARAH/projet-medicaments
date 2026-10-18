import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml

def manage_registry():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    model_name = params["model"]["model_name"]
    
    print(f"=== GESTION DU REGISTRE : {model_name} ===")
    
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        print(f"Version: {v.version} | Status: {v.status} | Aliases: {v.aliases} | Run ID: {v.run_id}")
        
    v_prod = "1"
    
    # Affectation de l'alias production
    client.set_registered_model_alias(name=model_name, alias="production", version=v_prod)
    
    client.update_model_version(
        name=model_name,
        version=v_prod,
        description="Modèle optimal de classification Princeps vs Générique (Alias Production)"
    )
    
    # Prédiction via l'Alias Production
    model_uri = f"models:/{model_name}@production"
    print(f"\nChargement depuis l'URI : {model_uri}")
    model_prod = mlflow.pyfunc.load_model(model_uri)
    
    sample_data = pd.DataFrame({
        "PPV": [15.0, 120.0],
        "PH": [10.0, 85.0],
        "PRIX_BR": [15.0, 100.0],
        "TAUX_REMBOURSEMENT_NUM": [70.0, 70.0],
        "DOSAGE_NUM": [500.0, 1000.0],
        "NB_PRESENTATIONS_DCI": [12.0, 1.0],
        "PPV_MOYEN_DCI": [18.0, 125.0],
        "RATIO_PPV_BR": [1.0, 1.2],
        "DIFF_PPV_PH": [5.0, 35.0]
    })
    
    preds = model_prod.predict(sample_data)
    print("Prédiction (0 = Princeps, 1 = Générique) :", preds)

if __name__ == "__main__":
    manage_registry()