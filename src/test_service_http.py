import requests
import json

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def test_http_service():
    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}

    # ---------------------------------------------------------
    # TEST 1 : REQUÊTE VALIDE (1 Générique + 1 Princeps)
    # ---------------------------------------------------------
    print_section("TEST 1 : REQUÊTE CONFORME (Test de Prédiction)")
    
    valid_payload = {
        "dataframe_split": {
            "columns": [
                "PPV", "PH", "PRIX_BR", "TAUX_REMBOURSEMENT_NUM",
                "DOSAGE_NUM", "NB_PRESENTATIONS_DCI", "PPV_MOYEN_DCI",
                "RATIO_PPV_BR", "DIFF_PPV_PH"
            ],
            "data": [
                # Échantillon 1 : Médicament Générique (ex: Prix bas, nombreuses présentations DCI)
                [15.0, 10.0, 15.0, 70.0, 500.0, 12.0, 18.0, 1.0, 5.0],
                # Échantillon 2 : Médicament Princeps (ex: Prix élevé, présentation unique)
                [350.0, 250.0, 300.0, 70.0, 100.0, 1.0, 350.0, 1.16, 100.0]
            ]
        }
    }

    try:
        response = requests.post(url, data=json.dumps(valid_payload), headers=headers)
        print(f"--> Code HTTP   : {response.status_code} OK")
        print("--> Résultat    :")
        print(json.dumps(response.json(), indent=2))
        print("\nInterpretation : 1 = Générique, 0 = Princeps")
    except Exception as e:
        print("Erreur de connexion au serveur :", e)

    # ---------------------------------------------------------
    # TEST 2 : REQUÊTE INVALIDE (Garde-fou / Contrôle de schéma)
    # ---------------------------------------------------------
    print_section("TEST 2 : REQUÊTE INVALIDE (Contrôle de Schéma)")
    
    invalid_payload = {
        "dataframe_split": {
            "columns": ["PPV", "PH"],  # Colonnes obligatoires manquantes
            "data": [[15.0, 10.0]]
        }
    }

    try:
        response_invalid = requests.post(url, data=json.dumps(invalid_payload), headers=headers)
        print(f"--> Code HTTP   : {response_invalid.status_code} (Rejeté par le serveur)")
        print("--> Message     :")
        
        # Formatage propre du message d'erreur
        err_json = response_invalid.json()
        print(f"    Code d'erreur : {err_json.get('error_code')}")
        print(f"    Classe        : {err_json.get('error_class')}")
        print(f"    Détail        : Schéma incomplet (champs requis manquants)")
    except Exception as e:
        print("Erreur de connexion au serveur :", e)

if __name__ == "__main__":
    test_http_service()