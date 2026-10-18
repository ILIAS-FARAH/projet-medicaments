import yaml
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def train_step():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    train_df = pd.read_csv(params["data"]["train_path"])
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    
    algo_name = params["model"]["selected_algorithm"]
    algo_params = params["algorithms"].get(algo_name, {})
    
    if algo_name == "gradient_boosting":
        model = GradientBoostingClassifier(**algo_params)
    elif algo_name == "decision_tree":
        model = DecisionTreeClassifier(**algo_params)
    elif algo_name == "knn":
        model = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(**algo_params))])
    elif algo_name == "logistic_regression":
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**algo_params))])
    elif algo_name == "mlp":
        model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(**algo_params))])
    else:
        raise ValueError(f"Algorithme non reconnu : {algo_name}")
        
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, params["model"]["artifact_path"])
    print(f"[Etape 2 - Train] Modèle '{algo_name}' entraîné et sauvegardé.")

if __name__ == "__main__":
    train_step()