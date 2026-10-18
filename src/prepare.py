import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_prep import prepare_data

def prepare_step():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    X, y = prepare_data(params["data"]["raw_path"])
    df = X.copy()
    df["target"] = y
    
    train_df, test_df = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=df["target"]
    )
    
    os.makedirs("data/prepare", exist_ok=True)
    train_df.to_csv(params["data"]["train_path"], index=False)
    test_df.to_csv(params["data"]["test_path"], index=False)
    
    print(f"[Etape 1 - Prepare] Train: {len(train_df)} lignes, Test: {len(test_df)} lignes")

if __name__ == "__main__":
    prepare_step()