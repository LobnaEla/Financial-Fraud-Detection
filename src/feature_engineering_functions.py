import pandas as pd
import joblib
import datetime
import numpy as np
import hashlib
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import datetime

# --------------- 1. Feature engineering transaction ---------------


def categorize_hour_of_day(hour):
    if hour > 3 and hour < 11:
        return 3
    if hour == 11 or hour == 18:
        return 1
    if hour == 2 or hour == 3 or hour == 23:
        return 2
    else:
        return 0


def log_transform_transaction_amt(transaction):
    transaction["TransactionAmt"] = np.log1p(transaction["TransactionAmt"])
    return transaction


def outlier(transaction):
    with open("src\outlier_bounds.pkl", "rb") as f:
        outlier_bounds = pickle.load(f)
    for col, (lower, upper) in outlier_bounds.items():
        if col in transaction.columns:
            transaction[col + "_is_outlier"] = (
                (transaction[col] < lower) | (transaction[col] > upper)
            ).astype("int8")
        else:
            transaction[col + "_is_outlier"] = 0

    return transaction


def alertfeature_transaction(transaction):
    transaction["alertFeature"] = transaction["hour"].apply(categorize_hour_of_day)
    return transaction


def extract_device_info_from_transaction(transaction):
    # Apply per row
    transaction["device_name"] = transaction["DeviceInfo"].apply(
        lambda x: x.split("/")[0] if isinstance(x, str) else None
    )
    transaction["device_version"] = transaction["DeviceInfo"].apply(
        lambda x: x.split("/")[1] if isinstance(x, str) and "/" in x else None
    )
    transaction["OS_id_30"] = transaction["id_30"].apply(
        lambda x: x.split(" ")[0] if isinstance(x, str) else None
    )
    transaction["browser_id_31"] = transaction["id_31"].apply(
        lambda x: x.split(" ")[0] if isinstance(x, str) else None
    )

    # Remplacement de certaines valeurs spécifiques pour 'device_name'
    def standardize_device(name):
        if not isinstance(name, str):
            return None
        if "SM" in name or "SAMSUNG" in name or "GT-" in name:
            return "Samsung"
        elif "Moto G" in name or "Moto" in name or "moto" in name:
            return "Motorola"
        elif "LG-" in name:
            return "LG"
        elif "rv:" in name:
            return "RV"
        elif "HUAWEI" in name or "ALE-" in name or "-L" in name:
            return "Huawei"
        elif "Blade" in name or "BLADE" in name:
            return "ZTE"
        elif "Linux" in name:
            return "Linux"
        elif "XT" in name:
            return "Sony"
        elif "HTC" in name:
            return "HTC"
        elif "ASUS" in name:
            return "Asus"
        else:
            return name

    transaction["device_name"] = transaction["device_name"].apply(standardize_device)

    transaction["had_id"] = 1

    return transaction


def compute_transaction_date_features(transaction, start_date="2017-11-30"):
    # Conversion de START_DATE en datetime
    startdate = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    # Calcul de la date à partir de TransactionDT (en secondes depuis le start_date)
    transaction["TransactionDT_date"] = startdate + pd.to_timedelta(
        transaction["TransactionDT"], unit="s"
    )

    # Calcul de DT_D, DT_W et DT_M
    transaction["DT_D"] = (
        (transaction["TransactionDT_date"].dt.year - 2017) * 365
        + transaction["TransactionDT_date"].dt.dayofyear
    ).astype(np.int16)
    transaction["DT_W"] = (
        transaction["TransactionDT_date"].dt.year - 2017
    ) * 52 + transaction["TransactionDT_date"].dt.isocalendar().week
    transaction["DT_M"] = (
        transaction["TransactionDT_date"].dt.year - 2017
    ) * 12 + transaction["TransactionDT_date"].dt.month

    return transaction


def normalize_d_column_times(transaction):
    transaction_dt = transaction["TransactionDT"]
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        col = f"D{i}"
        if col in transaction and not transaction[col].isnull().all():
            transaction[col] = transaction[col] - transaction_dt / np.float32(
                24 * 60 * 60
            )
    return transaction


def label_encode_transaction(transaction):
    with open("src/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    for col, mapping in label_encoders.items():
        if col in transaction.columns:
            transaction[col] = transaction[col].apply(
                lambda x: mapping.get(x, -1)
            )  # Utilise -1 pour les valeurs inconnues

    return transaction


def missing(transaction, fill_value=-999):
    for key, value in transaction.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            transaction[key] = fill_value
    return transaction


from sklearn.preprocessing import LabelEncoder
import pickle

import pandas as pd


def apply_productcd_day_count(transaction):
    import pickle
    import pandas as pd

    with open("src/train_grouped.pkl", "rb") as f:
        grouped_data = pickle.load(f)

    # Créer un dictionnaire pour chaque ProductCD
    grouped_dict = {}
    for code in ["W", "C", "R", "H", "S"]:
        subset = grouped_data[grouped_data["ProductCD"] == code]
        grouped_dict[code] = dict(
            zip(zip(subset["ProductCD"], subset["DT_D"]), subset["isFraud"])
        )

    # Extraire les valeurs scalaires
    product_cd = transaction["ProductCD"]
    dt_d = transaction["DT_D"]

    if isinstance(product_cd, pd.Series):
        product_cd = product_cd.item()
    if isinstance(dt_d, pd.Series):
        dt_d = dt_d.item()

    # Ajouter les colonnes correspondantes
    for code in ["W", "C", "R", "H", "S"]:
        key = (code, dt_d)
        if product_cd == code:
            transaction[f"ProductCD_{code}_Day"] = grouped_dict.get(code, {}).get(
                key, 0
            )
        else:
            transaction[f"ProductCD_{code}_Day"] = 0

    return transaction


def generate_device_hash_for_transaction(transaction):
    features = ["id_30", "id_31", "id_32", "id_33", "DeviceType", "DeviceInfo"]
    s = "".join(str(transaction.get(f, "")) for f in features)
    transaction["device_hash"] = hashlib.sha256(s.encode("utf-8")).hexdigest()[:15]
    return transaction


def compute_decimal_digit_for_transaction(transaction):
    amt = transaction.get("TransactionAmt", 0)
    amt = np.round(amt, 3)
    num = 3
    dec = int(np.round(amt * 1000))
    while dec % 10 == 0 and num > 0:
        num -= 1
        dec = dec // 10
    transaction["decimal_digit"] = num
    return transaction


def generate_device_counts_for_transaction(transaction):
    # Charger les mappings depuis le fichier
    with open("src/device_counts_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    # Récupérer les mappings pour uid et device_hash
    tmp_uid_device = mappings["uid_device_counts"]
    tmp_device_uid = mappings["device_uid_counts"]

    # Extraire le uid et device_hash de la transaction
    uid = transaction["uid"]
    device_hash = transaction["device_hash"]

    # Calcul des valeurs pour cette transaction spécifique
    transaction["uid_device_nunique"] = tmp_uid_device.get(
        uid, 0
    )  # Si pas trouvé, retourner 0
    transaction["device_uid_nunique"] = tmp_device_uid.get(
        device_hash, 0
    )  # Si pas trouvé, retourner 0

    return transaction


def generate_additional_transaction_features(transaction):
    if "had_id" in transaction.columns:
        transaction["had_id"] = transaction["had_id"].fillna(0)
    transaction["dow"] = transaction["TransactionDT_date"].dt.weekday
    transaction["hour"] = transaction["TransactionDT_date"].dt.hour
    transaction["email_domain_comp"] = (
        transaction["P_emaildomain"] == transaction["R_emaildomain"]
    ).astype(int)

    return transaction


def add_day_hour_counts(transaction):
    import pickle

    # Charger les mappings sauvegardés
    with open("src/day_hour_counts.pkl", "rb") as f:
        mappings = pickle.load(f)

    day_mapping = mappings["day_count"]
    hour_mapping = mappings["hour_count"]

    # Extraire les clefs à partir de la date pour une seule ligne
    date_obj = transaction["TransactionDT_date"].iloc[
        0
    ]  # Accéder à la première valeur (scalaire)

    # Vérifier que date_obj est bien de type datetime
    if isinstance(date_obj, pd.Timestamp):
        day_key = date_obj.date()
        hour_key = date_obj.strftime("%Y-%m-%d %H")

        # Ajouter les features
        transaction["day_count"] = day_mapping.get(day_key, 0)
        transaction["hour_count"] = hour_mapping.get(hour_key, 0)
    else:
        # Si la valeur n'est pas un Timestamp, on peut mettre des valeurs par défaut
        transaction["day_count"] = 0
        transaction["hour_count"] = 0

    return transaction


def apply_count_encoding_to_transaction(transaction):
    import pickle
    import numpy as np

    # Charger les mappings depuis le fichier
    with open("src/count_mappings.pkl", "rb") as f:
        count_mappings = pickle.load(f)

    for col, mapping in count_mappings.items():
        # Récupérer la valeur dans la colonne pour cette ligne spécifique
        value = transaction[col].iloc[0] if col in transaction else np.nan

        # Appliquer le count encoding
        transaction[col + "_count_full"] = mapping.get(value, 0)

    return transaction


def clean_transaction(transaction):
    import pickle

    # Charger les colonnes à supprimer depuis le fichier pickle
    with open("src/drop_columns.pkl", "rb") as f:
        drop_cols = pickle.load(f)

    # Ajouter les colonnes à supprimer manuellement
    drop_cols += [
        "TransactionID",
        "device_hash",
        "TransactionDT",
        "TransactionDT_date",
        "uid",
    ]

    # Supprimer les colonnes du DataFrame
    transaction = transaction.drop(columns=drop_cols, errors="ignore")

    return transaction


def apply_common_values_to_transaction(transaction):
    import pickle
    import numpy as np

    # Charger les valeurs communes sauvegardées
    with open("src/common_values.pkl", "rb") as f:
        common_values = pickle.load(f)

    # Pour chaque colonne dans 'cat', appliquer la transformation
    for column, common_set in common_values.items():
        # Utiliser .iloc[0] pour obtenir la première valeur scalaire
        value = transaction[column].iloc[0] if column in transaction else np.nan

        # Vérifier si la valeur est dans le set des valeurs communes
        if value not in common_set:
            transaction[column] = -999  # Remplacer les valeurs non communes par -999

    return transaction


def compare_and_clean_columns(transaction_df):
    import pickle

    # Charger les colonnes attendues
    with open("src/ordered_columns.pkl", "rb") as f:
        expected_columns = pickle.load(f)

    input_columns = list(transaction_df.columns)

    # Supprimer les colonnes dupliquées
    transaction_df = transaction_df.loc[:, ~transaction_df.columns.duplicated()]

    # Identifier les colonnes manquantes et inattendues
    missing_cols = list(set(expected_columns) - set(input_columns))
    extra_cols = list(set(input_columns) - set(expected_columns))

    # LOG: pour investiguer
    print("\n=== DEBUG: Columns ===")
    print(f"Missing columns ({len(missing_cols)}): {missing_cols}")
    print(f"Extra columns ({len(extra_cols)}): {extra_cols}")

    # NE PAS ajouter les colonnes manquantes ici — laisse le modèle planter si elles sont essentielles
    # transaction_df = transaction_df.reindex(columns=expected_columns, fill_value=-999)

    # Supprimer les colonnes inattendues uniquement
    transaction_df.drop(columns=extra_cols, inplace=True, errors="ignore")

    # Réordonner les colonnes (sans forcer les manquantes)
    ordered_columns = [col for col in expected_columns if col in transaction_df.columns]
    transaction_df = transaction_df[ordered_columns]

    return transaction_df, {
        "missing_columns": missing_cols,
        "removed_unexpected_columns": extra_cols,
    }


def add_features(transaction):
    transaction = transaction.copy()
    transaction["open_card"] = transaction["DT_D"] - transaction["D1"]
    transaction["first_tran"] = transaction["DT_D"] - transaction["D2"]
    return transaction


import pickle
import hashlib


def process_product_id_for_transaction(transaction):
    # Charger le LabelEncoder et le count mapping
    with open("src/productid_all.pkl", "rb") as f:
        productid_data = pickle.load(f)

    le = productid_data["label_encoder"]
    count_mapping = productid_data["count_mapping"]

    # Construire la clé combinée
    combined = str(transaction["TransactionAmt"]) + "_" + str(transaction["ProductCD"])

    # Transformer en ProductID encodé
    if combined in le.classes_:
        product_id = le.transform([combined])[0]
    else:
        product_id = -1  # ou un code spécial si inconnu
    transaction["ProductID"] = product_id

    # Ajouter le count
    transaction["ProductID_count_full"] = count_mapping.get(product_id, 0)

    return transaction


def apply_cross_stats(transaction):
    # Charger les mappings sauvegardés
    with open("src/cross_stats.pkl", "rb") as f:
        cross_stats = pickle.load(f)

    # Définir les features continues et catégorielles utilisées dans le mapping
    con_fea = [
        "V258",
        "C1",
        "C14",
        "C13",
        "TransactionAmt",
        "D15",
        "D2",
        "id_02",
        "dist1",
        "V294",
        "C11",
    ]
    cat_fea = [
        "card1",
        "card2",
        "addr1",
        "card4",
        "R_emaildomain",
        "P_emaildomain",
        "ProductID",
        "uid",
    ]

    # Appliquer les moyennes et écarts-types groupés
    for cont in con_fea:
        # Assurer que cont_val est bien un scalaire et non une série
        cont_val = transaction.get(cont, np.nan)
        if isinstance(cont_val, pd.Series):
            cont_val = cont_val.iloc[0]  # Si c'est une Series, prend la première valeur

        # Si cont_val vaut -999, on le remplace par NaN
        if cont_val == -999:
            cont_val = np.nan

        for cat in cat_fea:
            cat_val = transaction.get(cat, np.nan)
            if isinstance(cat_val, pd.Series):
                cat_val = cat_val.iloc[0]  # Assurer que cat_val est un scalaire

            mean_key = f"{cont}_{cat}_mean"
            std_key = f"{cont}_{cat}_std"

            # Appliquer les stats de cross-mapping
            transaction[mean_key] = cross_stats.get(mean_key, {}).get(cat_val, np.nan)
            transaction[std_key] = cross_stats.get(std_key, {}).get(cat_val, np.nan)

    return transaction


def apply_crossover_features_to_transaction(transaction):
    # Charger les objets sauvegardés
    with open("src/crossover_data.pkl", "rb") as f:
        data = pickle.load(f)

    label_encoders = data["label_encoders"]
    cross_counts = data["cross_counts"]

    # Liste des combinaisons à traiter (doit être cohérente avec celle utilisée à l'entraînement)
    temp = [
        "DeviceInfo__P_emaildomain",
        "card1__card5",
        "card2__id_20",
        "card5__P_emaildomain",
        "addr1__card1",
        "addr1__addr2",
        "card1__card2",
        "card2__addr1",
        "card1__P_emaildomain",
        "card2__P_emaildomain",
        "addr1__P_emaildomain",
        "DeviceInfo__id_31",
        "DeviceInfo__id_20",
        "DeviceType__id_31",
        "DeviceType__id_20",
        "DeviceType__P_emaildomain",
        "card1__M4",
        "card2__M4",
        "addr1__M4",
        "P_emaildomain__M4",
        "uid__ProductID",
        "uid__DeviceInfo",
    ]

    # Appliquer concaténation + encodage + count
    for feature in temp:
        f1, f2 = feature.split("__")
        val = str(transaction.get(f1, "nan")) + "_" + str(transaction.get(f2, "nan"))

        # Encoder avec le label encoder correspondant
        le = label_encoders.get(feature)
        if le is not None and val in le.classes_:
            encoded_val = le.transform([val])[0]
        else:
            encoded_val = -1  # valeur inconnue

        transaction[feature] = encoded_val

        # Ajouter la feature "_count_full"
        count_dict = cross_counts.get(feature, {})
        transaction[feature + "_count_full"] = count_dict.get(encoded_val, 0)

    return transaction


def label_encode_transaction_categorical_features(transaction, train_df, test_df):
    for f in transaction.index:
        # Vérifie si la colonne est catégorielle
        if transaction[f].dtype == "object" or isinstance(
            transaction[f], pd.Categorical
        ):
            # Combine train et test pour s'assurer que toutes les catégories sont prises en compte
            df_comb = pd.concat([train_df[f], test_df[f]], axis=0)
            df_comb, _ = df_comb.factorize(sort=True)

            # Si le nombre de catégories dépasse 32000, utilise un type int32
            if df_comb.max() > 32000:
                print(f"{f} needs int32")
                transaction[f] = df_comb[
                    df_comb.index.get_loc(transaction.name)
                ].astype("int32")
            else:
                transaction[f] = df_comb[
                    df_comb.index.get_loc(transaction.name)
                ].astype("int16")

    return transaction


def coding(transaction):
    # Charger les mappings
    with open("src/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    FE_mappings = mappings["frequency_encoding"]
    LE_mappings = mappings["label_encoding"]
    AGG_mappings = mappings["aggregation_encoding"]

    # 1. Feature "cents"
    transaction["cents"] = (
        transaction["TransactionAmt"] - np.floor(transaction["TransactionAmt"])
    ).astype("float32")

    # 2. Combinaison de colonnes
    if "card1" in transaction.columns and "addr1" in transaction.columns:
        transaction["card1_addr1"] = (
            transaction["card1"].astype(str) + "_" + transaction["addr1"].astype(str)
        )
        transaction["card1_addr1_P_emaildomain"] = (
            transaction["card1_addr1"].astype(str)
            + "_"
            + transaction["P_emaildomain"].astype(str)
        )

    # 3. Frequency Encoding
    for col_FE, mapping in FE_mappings.items():
        original_col = col_FE.replace("_FE", "")
        if original_col in transaction.columns:
            transaction[col_FE] = (
                transaction[original_col].map(mapping).fillna(-999).astype("float32")
            )
    # 4. Label Encoding
    for col_LE, uniques in LE_mappings.items():
        if col_LE in transaction.columns:
            unique_mapping = {v: i for i, v in enumerate(uniques)}
            transaction[col_LE] = (
                transaction[col_LE].map(unique_mapping).fillna(-1).astype("int32")
            )

    # 5. Group Aggregations
    for col_AG, mapping in AGG_mappings.items():
        base_col = col_AG.split("_")[
            1
        ]  # Exemple : 'card1' dans 'TransactionAmt_card1_mean'
        if base_col in transaction.columns:
            transaction[col_AG] = (
                transaction[base_col].map(mapping).fillna(-999).astype("float32")
            )

    return transaction


def generate_transaction_specific_features(df):
    # Calcul du jour (day)
    df["day"] = df["TransactionDT"] / (24 * 60 * 60)

    # Calcul du 'uid' en combinant 'card1_addr1' et la différence entre 'day' et 'D1'
    df["uid"] = (
        df["card1_addr1"].astype(str) + "_" + np.floor(df["day"] - df["D1"]).astype(str)
    )
    return df


def coding2(transaction_df):

    with open("src/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    FE_mappings = mappings["frequency_encoding"]
    AGG_mappings = mappings["aggregation_encoding"]
    AGG2_mappings = mappings["AGG2_mappings"]

    one_transaction = transaction_df.copy()

    # 1. Frequency Encoding uniquement pour 'uid'
    if "uid_FE" in FE_mappings:
        one_transaction["uid_FE"] = (
            one_transaction["uid"]
            .map(FE_mappings["uid_FE"])
            .fillna(-999)
            .astype("float32")
        )

    # 2. AGG TransactionAmt, D4, D9, D10, D15 sur uid (mean et std)
    columns_agg1 = ["TransactionAmt", "D4", "D9", "D10", "D15"]
    for col in columns_agg1:
        for agg in ["mean", "std"]:
            col_name = f"{col}_uid_{agg}"
            if col_name in AGG_mappings:
                one_transaction[col_name] = (
                    one_transaction["uid"]
                    .map(AGG_mappings[col_name])
                    .fillna(-999)
                    .astype("float32")
                )

    # 3. AGG C1-C14 sauf C3 sur uid (mean)
    for x in range(1, 15):
        if x != 3:
            col = f"C{x}"
            col_name = f"{col}_uid_mean"
            if col_name in AGG_mappings:
                one_transaction[col_name] = (
                    one_transaction["uid"]
                    .map(AGG_mappings[col_name])
                    .fillna(-999)
                    .astype("float32")
                )

    # 4. AGG M1-M9 sur uid (mean)
    for x in range(1, 10):
        col = f"M{x}"
        col_name = f"{col}_uid_mean"
        if col_name in AGG_mappings:
            one_transaction[col_name] = (
                one_transaction["uid"]
                .map(AGG_mappings[col_name])
                .fillna(-999)
                .astype("float32")
            )

    # 5. AGG2 P_emaildomain, dist1, DT_M, id_02, cents sur uid (nunique count)
    columns_agg2_1 = ["P_emaildomain", "dist1", "DT_M", "id_02", "cents"]
    for col in columns_agg2_1:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = (
                one_transaction["uid"]
                .map(AGG2_mappings[new_col])
                .fillna(-999)
                .astype("float32")
            )

    # 6. AGG C14 sur uid (std)
    if "C14_uid_std" in AGG_mappings:
        one_transaction["C14_uid_std"] = (
            one_transaction["uid"]
            .map(AGG_mappings["C14_uid_std"])
            .fillna(-999)
            .astype("float32")
        )

    # 7. AGG2 C13, V314 sur uid (nunique count)
    for col in ["C13", "V314"]:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = (
                one_transaction["uid"]
                .map(AGG2_mappings[new_col])
                .fillna(-999)
                .astype("float32")
            )

    # 8. AGG2 V127, V136, V309, V307, V320 sur uid (nunique count)
    for col in ["V127", "V136", "V309", "V307", "V320"]:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = (
                one_transaction["uid"]
                .map(AGG2_mappings[new_col])
                .fillna(-999)
                .astype("float32")
            )

    # 9. Créer outsider15
    one_transaction["outsider15"] = (
        np.abs(one_transaction["D1"] - one_transaction["D15"]) > 3
    ).astype("int8")

    return one_transaction
