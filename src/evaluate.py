import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, ConfusionMatrixDisplay

def evaluate_step():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    test_df = pd.read_csv(params["data"]["test_path"])
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    model = joblib.load(params["model"]["artifact_path"])
    y_pred = model.predict(X_test)
    
    f1 = float(f1_score(y_test, y_pred, pos_label=1))
    precision = float(precision_score(y_test, y_pred, pos_label=1))
    recall = float(recall_score(y_test, y_pred, pos_label=1))
    
    metrics = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }
    
    os.makedirs("models/reports", exist_ok=True)
    
    with open("models/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Princeps", "Générique"], ax=ax, cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.savefig("models/reports/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    
    print(f"[Etape 3 - Evaluate] F1: {f1:.4f}, Précision: {precision:.4f}, Rappel: {recall:.4f}")
    return metrics

if __name__ == "__main__":
    evaluate_step()