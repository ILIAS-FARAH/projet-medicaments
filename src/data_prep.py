import pandas as pd
import numpy as np

def prepare_data(filepath: str):
    """
    Transforme le fichier Excel brut en variables numériques et cible.
    - Cible : PRINCEPS_GENERIQUE (1 = Générique, 0 = Princeps)
    - 9 variables numériques (dont 4 issues de l'ingénierie de variables)
    """
    df = pd.read_excel(filepath, engine="openpyxl")
    
    # Nettoyage et extraction de la cible
    df = df[df["PRINCEPS_GENERIQUE"].isin(["P", "G"])].copy()
    y = (df["PRINCEPS_GENERIQUE"] == "G").astype(int)
    
    # Variables numériques de base
    ppv = pd.to_numeric(df["PPV"], errors="coerce").fillna(0.0)
    ph = pd.to_numeric(df["PH"], errors="coerce").fillna(0.0)
    prix_br = pd.to_numeric(df["PRIX_BR"], errors="coerce").fillna(0.0)
    
    # Prise en compte du nom de colonne (avec ou sans underscore)
    taux_col = "TAUX_REMBOURSEMENT" if "TAUX_REMBOURSEMENT" in df.columns else "TAUX REMBOURSEMENT"
    
    # Conversion du taux de remboursement
    taux_remboursement = (
        df[taux_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    taux_remboursement_num = pd.to_numeric(taux_remboursement, errors="coerce").fillna(0.0)
    
    # Conversion du dosage
    dosage_raw = pd.to_numeric(df["DOSAGE1"], errors="coerce")
    dosage_num = dosage_raw.fillna(dosage_raw.median()).fillna(0.0)
    
    # Feature Engineering (Ingénierie de variables)
    dci_counts = df.groupby("DCI1")["CODE"].transform("count").fillna(1.0)
    dci_ppv_mean = df.groupby("DCI1")["PPV"].transform("mean").fillna(ppv)
    ratio_ppv_br = ppv / (prix_br + 1.0)
    diff_ppv_ph = ppv - ph
    
    # Assemblage des caractéristiques
    X = pd.DataFrame({
        "PPV": ppv,
        "PH": ph,
        "PRIX_BR": prix_br,
        "TAUX_REMBOURSEMENT_NUM": taux_remboursement_num,
        "DOSAGE_NUM": dosage_num,
        "NB_PRESENTATIONS_DCI": dci_counts.astype(float),
        "PPV_MOYEN_DCI": dci_ppv_mean.astype(float),
        "RATIO_PPV_BR": ratio_ppv_br.astype(float),
        "DIFF_PPV_PH": diff_ppv_ph.astype(float)
    })
    
    X = X.astype(np.float64)
    return X, y

if __name__ == "__main__":
    X, y = prepare_data("data/brut/ref-des-medicaments-cnops-2014.xlsx")
    print(f"Données transformées : {X.shape[0]} lignes, {X.shape[1]} variables.")
    print("Répartition de la cible :", y.value_counts().to_dict())