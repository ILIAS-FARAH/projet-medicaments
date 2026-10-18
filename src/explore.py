import pandas as pd

def explore(filepath: str):
    df = pd.read_excel(filepath, engine="openpyxl")
    
    print("=== EXPLORATION DU FICHIER EXCEL BRUT ===")
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("\nColonnes :", list(df.columns))
    
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\nRépartition PRINCEPS_GENERIQUE :")
    print(df["PRINCEPS_GENERIQUE"].value_counts())
    
    print("\nStatistiques PPV :")
    print(df["PPV"].describe())

if __name__ == "__main__":
    explore("data/brut/ref-des-medicaments-cnops-2014.xlsx")