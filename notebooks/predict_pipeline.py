# Imports nécessaires
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import datetime
import numpy as np
import hashlib
from sklearn.preprocessing import LabelEncoder



def alertfeature_transaction(transaction):
    transaction['alertFeature'] = transaction['hour'].apply(categorize_hour_of_day)
    return 

def extract_device_info_from_transaction(transaction):
    transaction['device_name'] = transaction['DeviceInfo'].split('/')[0]
    transaction['device_version'] = transaction['DeviceInfo'].split('/')[1] if len(transaction['DeviceInfo'].split('/')) > 1 else None
    transaction['OS_id_30'] = transaction['id_30'].split(' ')[0] if isinstance(transaction['id_30'], str) else None
    transaction['browser_id_31'] = transaction['id_31'].split(' ')[0] if isinstance(transaction['id_31'], str) else None
    
    # Remplacement de certaines valeurs spécifiques pour 'device_name'
    if 'SM' in transaction['device_name'] or 'SAMSUNG' in transaction['device_name'] or 'GT-' in transaction['device_name']:
        transaction['device_name'] = 'Samsung'
    elif 'Moto G' in transaction['device_name'] or 'Moto' in transaction['device_name'] or 'moto' in transaction['device_name']:
        transaction['device_name'] = 'Motorola'
    elif 'LG-' in transaction['device_name']:
        transaction['device_name'] = 'LG'
    elif 'rv:' in transaction['device_name']:
        transaction['device_name'] = 'RV'
    elif 'HUAWEI' in transaction['device_name'] or 'ALE-' in transaction['device_name'] or '-L' in transaction['device_name']:
        transaction['device_name'] = 'Huawei'
    elif 'Blade' in transaction['device_name'] or 'BLADE' in transaction['device_name']:
        transaction['device_name'] = 'ZTE'
    elif 'Linux' in transaction['device_name']:
        transaction['device_name'] = 'Linux'
    elif 'XT' in transaction['device_name']:
        transaction['device_name'] = 'Sony'
    elif 'HTC' in transaction['device_name']:
        transaction['device_name'] = 'HTC'
    elif 'ASUS' in transaction['device_name']:
        transaction['device_name'] = 'Asus'
    
    # Si le device_name est trop rare, on le place dans 'Others'
    rare_devices = ['device_name_list_here']  # Par exemple : ['Device1', 'Device2']
    if transaction['device_name'] in rare_devices:
        transaction['device_name'] = 'Others'
    
    # Ajout de la feature 'had_id'
    transaction['had_id'] = 1
    
    return transaction

def compute_transaction_date_features(transaction, start_date='2017-11-30'):
    # Conversion de START_DATE en datetime
    startdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    
    # Calcul de la date à partir de TransactionDT
    transaction['TransactionDT_date'] = startdate + datetime.timedelta(seconds=transaction['TransactionDT'])
    
    # Calcul de DT_D, DT_W et DT_M
    transaction['DT_D'] = ((transaction['TransactionDT_date'].dt.year - 2017) * 365 + transaction['TransactionDT_date'].dt.dayofyear).astype(np.int16)
    transaction['DT_W'] = (transaction['TransactionDT_date'].dt.year - 2017) * 52 + transaction['TransactionDT_date'].dt.isocalendar().week
    transaction['DT_M'] = (transaction['TransactionDT_date'].dt.year - 2017) * 12 + transaction['TransactionDT_date'].dt.month
    
    return transaction

def normalize_d_column_times(transaction, transaction_dt):
    # Normalisation des colonnes D1 à D15
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        # Normalisation pour chaque colonne D
        d_column_name = 'D' + str(i)
        if d_column_name in transaction:
            transaction[d_column_name] = transaction[d_column_name] - transaction_dt / np.float32(24 * 60 * 60)
    
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

def encode_features_by_frequency(df, cols):
    for col in cols:
        vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
        vc[-999] = -999  # Assure que les valeurs manquantes sont traitées
        nm = col + '_FE'
        df[nm] = df[col].map(vc).astype('float32')
        print(nm, ', ', end='')

def label_encode_feature(col, df, verbose=True):
    df_comb, _ = pd.factorize(df[col], sort=True)
    nm = col
    if df_comb.max() > 32000:
        df[nm] = df_comb.astype('int32')
    else:
        df[nm] = df_comb.astype('int16')
    if verbose:
        print(nm, ', ', end='')

def aggregate_group_features(main_column, uids, x, aggregations=['mean'], fillna=True):
    for col in uids:
        for agg_type in aggregations:
            new_col_name = main_column + '_' + col + '_' + agg_type
            if agg_type == 'mean':
                value = x[main_column].mean()
            elif agg_type == 'std':
                value = x[main_column].std()
            x[new_col_name] = value
            if fillna:
                x[new_col_name].fillna(-999, inplace=True)
            print("'" + new_col_name + "'", ', ', end='')

def combine_and_label_encode_columns(col1, col2, df):
    nm = col1 + '_' + col2
    df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
    label_encode_feature(nm, df, verbose=False)
    print(nm, ', ', end='')

def calculate_transaction_cents(df):
    df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')
    print('cents, ', end='')

def apply_feature_transformations_to_transaction(df):
    # Calcul des centimes
    calculate_transaction_cents(df)
    
    # Encodage par fréquence sur les colonnes spécifiées
    encode_features_by_frequency(df, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
    
    # Combinaison de colonnes et encodage
    combine_and_label_encode_columns('card1', 'addr1', df)
    combine_and_label_encode_columns('card1_addr1', 'P_emaildomain', df)
    
    # Encodage par fréquence sur les nouvelles colonnes combinées
    encode_features_by_frequency(df, ['card1_addr1', 'card1_addr1_P_emaildomain'])
    
    # Agrégation des colonnes 'TransactionAmt', 'D9', 'D11' avec les agrégations 'mean' et 'std'
    aggregate_group_features('TransactionAmt', ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'], df)

def generate_transaction_specific_features(df):
    # Calcul du jour (day)
    df['day'] = df['TransactionDT'] / (24*60*60)
    
    # Calcul du 'uid' en combinant 'card1_addr1' et la différence entre 'day' et 'D1'
    df['uid'] = str(df['card1_addr1']) + '_' + str(np.floor(df['day'] - df['D1']))
    return df

def generate_device_hash_for_transaction(x):
    # Création de la chaîne de caractères à partir des colonnes id_30, id_31, id_32, id_33, DeviceType, et DeviceInfo
    s = str(x['id_30']) + str(x['id_31']) + str(x['id_32']) + str(x['id_33']) + str(x['DeviceType']) + str(x['DeviceInfo'])
    
    # Calcul du hash sha256 et extraction des 15 premiers caractères
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h

def generate_additional_transaction_features(x):
    # Subdiviser la date en jour de la semaine et heure
    x['dow'] = x['TransactionDT_date'].dayofweek  # Jour de la semaine (0=Monday, 6=Sunday)
    x['hour'] = x['TransactionDT_date'].hour      # Heure de la journée (0-23)

    # Comparaison entre les domaines des emails P_emaildomain et R_emaildomain
    x['email_domain_comp'] = (x['P_emaildomain'] == x['R_emaildomain']).astype(int)
    x['open_card'] = x.DT_D - x['D1']
    x['first_tran'] = x.DT_D - x['D2']
    return x

def categorize_hour_of_day(hour):
    if hour > 3 and hour < 11:
        return 3
    if hour == 11 or hour == 18:
        return 1
    if hour == 2 or hour == 3 or hour == 23:
        return 2
    else:
        return 0

def categorize_and_apply_hour_feature(train):
    train['alertFeature'] = train['hour'].apply(categorize_hour_of_day)
    
    return train

def productid(x):
    temp123 = ['TransactionAmt__ProductCD']
    for feature in temp123:
        f1, f2 = feature.split('__')
        x[feature] = x[f1].astype(str) + '_' + x[f2].astype(str)
    x.rename(columns = {'TransactionAmt__ProductCD':'ProductID'},inplace=True)
 
def clean_transaction(x):
    drop = ['DeviceInfo', 'device_version', 'DT_D', 'DT_W', 'DT_M', 'TransactionAmt_ProductID_mean',
            'C1_P_emaildomain_mean', 'id_17_count_full', 'id_02_R_emaildomain_std', 
            'C11_P_emaildomain_mean', 'TransactionAmt_R_emaildomain_std',
            'V258_card4_mean', 'id_18_count_full', 'id_02_R_emaildomain_mean', 
            'C1_R_emaildomain_mean', 'C13_R_emaildomain_std', 'browser_id_31', 
            'D15_R_emaildomain_std', 'C14_R_emaildomain_std', 'V58', 'C1_R_emaildomain_std', 
            'C11_P_emaildomain_std', 'C1_card4_mean', 'ProductCD', 'TransactionAmt_card4_mean', 
            'V258_card4_std', 'id_26_count_full', 'C1_card4_std', 'id_25_count_full',
            'dist1_card4_std', 'addr2_count_full', 'TransactionAmt_card4_std', 
            'C11_R_emaildomain_mean', 'D15_card4_std', 'DeviceType', 'dist1_card4_mean', 
            'id_21_count_full', 'C11_R_emaildomain_std', 'C14_card4_mean', 'C14_card4_std',
            'id_02_card4_mean', 'id_02_card4_std', 'C13_card4_mean', 'id_24_count_full',
            'D2_revised_card4_mean', 'had_id', 'D2_card4_std', 'TransactionAmt_ProductID_std',
            'id_22_count_full', 'C13_card4_std', 'V294_card4_std', 'V294_card4_mean', 
            'dist1_R_emaildomain_mean', 'C11_card4_mean', 'C11_card4_std', 
            'dist1_R_emaildomain_std', 'D15_card4_mean']
    
    drop1 = drop[:200]
    drop2 = drop[200:]

    for col_list in [drop1, drop2, 
                     ['TransactionID','TransactionDT','TransactionDT_date','uid'],
                     ['device_hash']]:
        x = x.drop(col_list, axis=0, errors='ignore')

    return x


def predict_with_stacking(x):
    # Charger les modèles
    xgb_model = joblib.load('/kaggle/working/xgb_model.pkl')
    catboost_model = joblib.load('/kaggle/working/cat_model.pkl')
    meta_model = joblib.load('/kaggle/working/meta_model_val.pkl')
    
    # 1. Prédictions individuelles
    xgb_pred = xgb_model.predict_proba(x)[:, 1]  # prédiction de la probabilité positive
    cat_pred = catboost_model.predict_proba(x)[:, 1]
    
    # 2. Préparer les features pour le métamodèle (LightGBM)
    meta_features = np.column_stack((xgb_pred, cat_pred))  # shape (1, 2)

    # 3. Prédiction finale avec le métamodèle
    final_pred = meta_model.predict(meta_features)
    
    return final_pred