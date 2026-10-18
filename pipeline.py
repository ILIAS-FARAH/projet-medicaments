import sys
import os
import json
import yaml
import mlflow
import joblib
import pandas as pd
from mlflow.models import infer_signature

from src.prepare import prepare_step
from src.train import train_step
from src.evaluate import evaluate_step

def run_pipeline():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Ne définit l'expérience que si la CLI MLflow ne l'a pas déjà initialisée
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment("medicaments-pipeline")
    
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    threshold = params["validation"]["f1_threshold"]
    algo_selected = params["model"]["selected_algorithm"]
    model_name = params["model"]["model_name"]
    
    active_run = mlflow.active_run()
    run_context = active_run if active_run else mlflow.start_run(run_name=f"pipeline_parent_{algo_selected}")
    
    with run_context as parent_run:
        # 1. Étape Préparation
        with mlflow.start_run(run_name="1_prepare", nested=True):
            prepare_step()
            
        # 2. Étape Entraînement
        with mlflow.start_run(run_name="2_train", nested=True):
            train_step()
            
        # 3. Étape Évaluation
        with mlflow.start_run(run_name="3_evaluate", nested=True):
            metrics = evaluate_step()
            
        # Log des paramètres et métriques sur le Run Parent
        mlflow.log_params(params["algorithms"][algo_selected])
        mlflow.log_param("algorithm", algo_selected)
        mlflow.log_param("f1_threshold_required", threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("models/reports/confusion_matrix.png")
        
        # Vérification de la Règle de Validation
        f1_obtained = metrics["f1_score"]
        
        if f1_obtained >= threshold:
            print(f"\n✅ VALIDE : F1 obtenu ({f1_obtained:.4f}) >= Seuil ({threshold:.4f})")
            mlflow.set_tag("validation_status", "ACCEPTED")
            
            # Signature du modèle
            test_df = pd.read_csv(params["data"]["test_path"])
            X_test = test_df.drop(columns=["target"])
            loaded_model = joblib.load(params["model"]["artifact_path"])
            
            predictions = loaded_model.predict(X_test)
            signature = infer_signature(X_test, predictions)
            input_example = X_test.iloc[:2]
            
            # Enregistrement au Registre MLflow avec autorisation du type skops
            mlflow.sklearn.log_model(
                sk_model=loaded_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name,
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            print(f"Modèle enregistré sous le nom '{model_name}' dans le Registre MLflow.")
        else:
            print(f"\n❌ REFUSÉ : F1 obtenu ({f1_obtained:.4f}) < Seuil ({threshold:.4f})")
            mlflow.set_tag("validation_status", "REFUSED")
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()