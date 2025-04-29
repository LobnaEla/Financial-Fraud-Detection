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
    transaction['TransactionAmt'] = np.log1p(transaction['TransactionAmt'])
    return transaction

def outlier(transaction):
    with open('src\outlier_bounds.pkl', 'rb') as f:
        outlier_bounds = pickle.load(f)
    for col, (lower, upper) in outlier_bounds.items():
        if col in transaction.columns:
            transaction[col + '_is_outlier'] = ((transaction[col] < lower) | (transaction[col] > upper)).astype('int8')
        else:
            transaction[col + '_is_outlier'] = 0

    return transaction


def alertfeature_transaction(transaction):
    transaction['alertFeature'] = transaction['hour'].apply(categorize_hour_of_day)
    return transaction

def extract_device_info_from_transaction(transaction):
    # Apply per row
    transaction['device_name'] = transaction['DeviceInfo'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else None)
    transaction['device_version'] = transaction['DeviceInfo'].apply(lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else None)
    transaction['OS_id_30'] = transaction['id_30'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else None)
    transaction['browser_id_31'] = transaction['id_31'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else None)
    
    # Remplacement de certaines valeurs spécifiques pour 'device_name'
    def standardize_device(name):
        if not isinstance(name, str):
            return None
        if 'SM' in name or 'SAMSUNG' in name or 'GT-' in name:
            return 'Samsung'
        elif 'Moto G' in name or 'Moto' in name or 'moto' in name:
            return 'Motorola'
        elif 'LG-' in name:
            return 'LG'
        elif 'rv:' in name:
            return 'RV'
        elif 'HUAWEI' in name or 'ALE-' in name or '-L' in name:
            return 'Huawei'
        elif 'Blade' in name or 'BLADE' in name:
            return 'ZTE'
        elif 'Linux' in name:
            return 'Linux'
        elif 'XT' in name:
            return 'Sony'
        elif 'HTC' in name:
            return 'HTC'
        elif 'ASUS' in name:
            return 'Asus'
        else:
            return name

    transaction['device_name'] = transaction['device_name'].apply(standardize_device)

    transaction['had_id'] = 1

    return transaction


def compute_transaction_date_features(transaction, start_date='2017-11-30'):
    # Conversion de START_DATE en datetime
    startdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    
    # Calcul de la date à partir de TransactionDT (en secondes depuis le start_date)
    transaction['TransactionDT_date'] = startdate + pd.to_timedelta(transaction['TransactionDT'], unit='s')
    
    # Calcul de DT_D, DT_W et DT_M
    transaction['DT_D'] = ((transaction['TransactionDT_date'].dt.year - 2017) * 365 + transaction['TransactionDT_date'].dt.dayofyear).astype(np.int16)
    transaction['DT_W'] = (transaction['TransactionDT_date'].dt.year - 2017) * 52 + transaction['TransactionDT_date'].dt.isocalendar().week
    transaction['DT_M'] = (transaction['TransactionDT_date'].dt.year - 2017) * 12 + transaction['TransactionDT_date'].dt.month
    
    return transaction


def normalize_d_column_times(transaction):
    transaction_dt = transaction['TransactionDT']
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        col = f'D{i}'
        if col in transaction and not transaction[col].isnull().all():  # vérifier si toute la colonne n'est pas NaN
            transaction[col] = transaction[col] - transaction_dt / np.float32(24 * 60 * 60)
    return transaction


def label_encode_transaction(transaction):
    with open('src/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    for col, mapping in label_encoders.items():
        if col in transaction.columns:
            transaction[col] = transaction[col].apply(lambda x: mapping.get(x, -1))  # Utilise -1 pour les valeurs inconnues

    return transaction



def missing(transaction, fill_value=-999):
    for key, value in transaction.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            transaction[key] = fill_value
    return transaction

from sklearn.preprocessing import LabelEncoder
import pickle

import pickle

import pandas as pd
import pickle

import pandas as pd
import pickle

def productid(transaction, label_encoder_path='src/productid_labelencoder.pkl'):
    if transaction is None :
        raise ValueError("L'entrée doit être une ligne (pd.Series) non nulle.")
    
    # Conversion des types de données pour éviter les erreurs avec les modèles
    # Convertir les dates en timestamp (secondes depuis 1970-01-01)
    if 'TransactionDT_date' in transaction and isinstance(transaction['TransactionDT_date'], pd.Timestamp):
        transaction['TransactionDT_date'] = transaction['TransactionDT_date'].timestamp()

    # Convertir les colonnes 'uid' et 'device_hash' en catégories (ou les encoder si nécessaire)
    if 'uid' in transaction and isinstance(transaction['uid'], str):
        transaction['uid'] = hash(transaction['uid'])  # Ou utiliser LabelEncoder si nécessaire
    if 'device_hash' in transaction and isinstance(transaction['device_hash'], str):
        transaction['device_hash'] = hash(transaction['device_hash'])  # Ou LabelEncoder aussi

    # Récupérer les valeurs de manière sûre
    f1 = str(transaction['TransactionAmt']) if pd.notnull(transaction.get('TransactionAmt')) else ''
    f2 = str(transaction['ProductCD']) if pd.notnull(transaction.get('ProductCD')) else ''

    combined_value = f1 + '_' + f2
    transaction['ProductID_raw'] = combined_value

    # Charger le label encoder
    with open(label_encoder_path, 'rb') as f:
        le = pickle.load(f)

    # Encoder la combinaison
    if combined_value in le.classes_:
        transaction['ProductID'] = le.transform([combined_value])[0]
    else:
        transaction['ProductID'] = -1

    return transaction


def generate_device_hash_for_transaction(transaction):
    features = ['id_30', 'id_31', 'id_32', 'id_33', 'DeviceType', 'DeviceInfo']
    s = ''.join(str(transaction.get(f, '')) for f in features)
    transaction['device_hash'] = hashlib.sha256(s.encode('utf-8')).hexdigest()[:15]
    return transaction

    
def generate_additional_transaction_features(transaction):
    transaction['dow'] = transaction['TransactionDT_date'].dt.weekday
    transaction['hour'] = transaction['TransactionDT_date'].dt.hour
    transaction['email_domain_comp'] = (transaction['P_emaildomain'] == transaction['R_emaildomain']).astype(int)
    
    if 'had_id' in transaction.columns:
        transaction['had_id'] = transaction['had_id'].fillna(0)
        
    return transaction


def clean_transaction(transaction):
    drop_cols = ['DeviceInfo', 'device_version', 'DT_D', 'DT_W', 'DT_M', 
                 'TransactionID', 'TransactionDT', 'TransactionDT_date', 'uid',
                 'device_hash', 'DeviceType', 'browser_id_31']
    
    # Utilisation de drop pour supprimer les colonnes
    transaction = transaction.drop(columns=drop_cols, errors='ignore')  # 'ignore' évite les erreurs si certaines colonnes n'existent pas
    
    return transaction


def add_features(transaction):
    transaction = transaction.copy()
    transaction['open_card'] = transaction['DT_D'] - transaction['D1']
    transaction['first_tran'] = transaction['DT_D'] - transaction['D2']
    return transaction



def label_encode_transaction_categorical_features(transaction, train_df, test_df):
    for f in transaction.index:
        # Vérifie si la colonne est catégorielle
        if transaction[f].dtype == 'object' or isinstance(transaction[f], pd.Categorical):
            # Combine train et test pour s'assurer que toutes les catégories sont prises en compte
            df_comb = pd.concat([train_df[f], test_df[f]], axis=0)
            df_comb, _ = df_comb.factorize(sort=True)
            
            # Si le nombre de catégories dépasse 32000, utilise un type int32
            if df_comb.max() > 32000:
                print(f"{f} needs int32")
                transaction[f] = df_comb[df_comb.index.get_loc(transaction.name)].astype('int32')
            else:
                transaction[f] = df_comb[df_comb.index.get_loc(transaction.name)].astype('int16')

    return transaction

def coding(df):
    # Charger les mappings
    with open('src/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    FE_mappings = mappings['frequency_encoding']
    LE_mappings = mappings['label_encoding']
    AGG_mappings = mappings['aggregation_encoding']

    # 1. Feature "cents"
    df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')

    # 2. Frequency Encoding
    for col_FE, mapping in FE_mappings.items():
        original_col = col_FE.replace('_FE', '')  # Assurez-vous que la colonne originale existe
        if original_col in df.columns:
            df[col_FE] = df[original_col].map(mapping).fillna(-999).astype('float32')

    # 3. Combinaison de colonnes
    if 'card1' in df.columns and 'addr1' in df.columns:
        df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        df['card1_addr1_P_emaildomain'] = df['card1_addr1'].astype(str) + '_' + df['P_emaildomain'].astype(str)

    # 4. Label Encoding
    for col_LE, uniques in LE_mappings.items():
        if col_LE in df.columns:
            unique_mapping = {v: i for i, v in enumerate(uniques)}
            df[col_LE] = df[col_LE].map(unique_mapping).fillna(-1).astype('int32')

    # 5. Group Aggregations
    for col_AG, mapping in AGG_mappings.items():
        base_col = col_AG.split('_')[1]  # Exemple : 'card1' dans 'TransactionAmt_card1_mean'
        if base_col in df.columns:
            df[col_AG] = df[base_col].map(mapping).fillna(-999).astype('float32')

    return df

def generate_transaction_specific_features(df):
    # Calcul du jour (day)
    df['day'] = df['TransactionDT'] / (24*60*60)
    
    # Calcul du 'uid' en combinant 'card1_addr1' et la différence entre 'day' et 'D1'
    df['uid'] = str(df['card1_addr1']) + '_' + str(np.floor(df['day'] - df['D1']))
    return df



def coding2(transaction_df):
  
    with open('src/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    FE_mappings = mappings['frequency_encoding']
    AGG_mappings = mappings['aggregation_encoding']
    AGG2_mappings = mappings['AGG2_mappings']

    one_transaction = transaction_df.copy()

    # 1. Frequency Encoding uniquement pour 'uid'
    if 'uid_FE' in FE_mappings:
        one_transaction['uid_FE'] = one_transaction['uid'].map(FE_mappings['uid_FE']).astype('float32')
    
    # 2. AGG TransactionAmt, D4, D9, D10, D15 sur uid (mean et std)
    columns_agg1 = ['TransactionAmt', 'D4', 'D9', 'D10', 'D15']
    for col in columns_agg1:
        for agg in ['mean', 'std']:
            col_name = f"{col}_uid_{agg}"
            if col_name in AGG_mappings:
                one_transaction[col_name] = one_transaction['uid'].map(AGG_mappings[col_name]).astype('float32')

    # 3. AGG C1-C14 sauf C3 sur uid (mean)
    for x in range(1, 15):
        if x != 3:
            col = f"C{x}"
            col_name = f"{col}_uid_mean"
            if col_name in AGG_mappings:
                one_transaction[col_name] = one_transaction['uid'].map(AGG_mappings[col_name]).astype('float32')

    # 4. AGG M1-M9 sur uid (mean)
    for x in range(1, 10):
        col = f"M{x}"
        col_name = f"{col}_uid_mean"
        if col_name in AGG_mappings:
            one_transaction[col_name] = one_transaction['uid'].map(AGG_mappings[col_name]).astype('float32')

    # 5. AGG2 P_emaildomain, dist1, DT_M, id_02, cents sur uid (nunique count)
    columns_agg2_1 = ['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents']
    for col in columns_agg2_1:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = one_transaction['uid'].map(AGG2_mappings[new_col]).astype('float32')

    # 6. AGG C14 sur uid (std)
    if 'C14_uid_std' in AGG_mappings:
        one_transaction['C14_uid_std'] = one_transaction['uid'].map(AGG_mappings['C14_uid_std']).astype('float32')

    # 7. AGG2 C13, V314 sur uid (nunique count)
    for col in ['C13', 'V314']:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = one_transaction['uid'].map(AGG2_mappings[new_col]).astype('float32')

    # 8. AGG2 V127, V136, V309, V307, V320 sur uid (nunique count)
    for col in ['V127', 'V136', 'V309', 'V307', 'V320']:
        new_col = f"uid_{col}_ct"
        if new_col in AGG2_mappings:
            one_transaction[new_col] = one_transaction['uid'].map(AGG2_mappings[new_col]).astype('float32')

    # 9. Créer outsider15
    one_transaction['outsider15'] = (np.abs(one_transaction['D1'] - one_transaction['D15']) > 3).astype('int8')

    return one_transaction
