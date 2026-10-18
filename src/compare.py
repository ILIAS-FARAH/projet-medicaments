import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.data_prep import prepare_data

def run_comparison():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("medicaments-exploration")
    
    X, y = prepare_data("data/brut/ref-des-medicaments-cnops-2014.xlsx")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    classifiers = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision_Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "MLP_Neural_Net": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42))
        ])
    }
    
    print(f"{'Algorithme':<20} | {'F1 Score':<10} | {'Précision':<10} | {'Rappel':<10} | {'Durée (s)':<10}")
    print("-" * 65)
    
    for name, model in classifiers.items():
        with mlflow.start_run(run_name=name):
            start_time = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, pos_label=1)
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            
            mlflow.log_param("algorithm", name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("training_duration", duration)
            mlflow.set_tag("dataset", "CNOPS_2014")
            
            print(f"{name:<20} | {f1:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {duration:<10.4f}")

if __name__ == "__main__":
    run_comparison()