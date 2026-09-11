import random
import numpy as np
import pandas as pd


def train_test_split_manual(df, test_ratio=0.2, random_state=42, stratify_col=None):
   
    random.seed(random_state)
    np.random.seed(random_state)
    
    if stratify_col is not None and stratify_col in df.columns:
        train_indices = []
        test_indices = []
        
        classes = df[stratify_col].unique()
        for cls in classes:
            cls_indices = df[df[stratify_col] == cls].index.tolist()
            random.shuffle(cls_indices)
            
            split_point = int(len(cls_indices) * (1 - test_ratio))
            train_indices.extend(cls_indices[:split_point])
            test_indices.extend(cls_indices[split_point:])
            
        train_df = df.loc[train_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = df.loc[test_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        indices = list(range(len(df)))
        random.shuffle(indices)
        
        split_point = int(len(df) * (1 - test_ratio))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
    print(f"[INFO] Train/Test Split completed: {len(train_df)} Training samples, {len(test_df)} Test samples.")
    return train_df, test_df


def encode_categorical_features(train_df, test_df, categorical_cols):
    encoding_dict = {}
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
            
        unique_categories = sorted(train_df[col].dropna().unique().tolist())
        encoding_dict[col] = unique_categories

        if len(unique_categories) == 2:
            val_0, val_1 = unique_categories[0], unique_categories[1]
            train_encoded[col] = (train_encoded[col] == val_1).astype(int)
            test_encoded[col] = (test_encoded[col] == val_1).astype(int)
        else:
            for cat in unique_categories:
                new_col_name = f"{col}_{cat}"
                train_encoded[new_col_name] = (train_encoded[col] == cat).astype(int)
                test_encoded[new_col_name] = (test_encoded[col] == cat).astype(int)
            train_encoded = train_encoded.drop(columns=[col])
            test_encoded = test_encoded.drop(columns=[col])
            
    return train_encoded, test_encoded, encoding_dict


def fit_minmax_scaler(X_train):
    
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values.astype(float)
    else:
        X_arr = np.array(X_train, dtype=float)
        
    mins = np.min(X_arr, axis=0)
    maxs = np.max(X_arr, axis=0)
    
    scaler_params = {
        'mins': mins,
        'maxs': maxs
    }
    return scaler_params


def transform_minmax_scaler(X, scaler_params):
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(float)
    else:
        X_arr = np.array(X, dtype=float)
        
    mins = scaler_params['mins']
    maxs = scaler_params['maxs']
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    
    X_scaled = (X_arr - mins) / ranges
    return X_scaled


def prepare_datasets(raw_df, target_col='G3', test_size=0.2, random_state=42):

    df = raw_df.copy()
    df['Pass_Status'] = (df['G3'] >= 10).astype(int)
    train_df, test_df = train_test_split_manual(
        df, test_ratio=test_size, random_state=random_state, stratify_col='Pass_Status'
    )
    y_train_reg = train_df['G3'].values.astype(float)
    y_test_reg = test_df['G3'].values.astype(float)
    
    y_train_clf = train_df['Pass_Status'].values.astype(int)
    y_test_clf = test_df['Pass_Status'].values.astype(int)
    drop_cols = ['G3', 'Pass_Status']
    train_features_df = train_df.drop(columns=drop_cols)
    test_features_df = test_df.drop(columns=drop_cols)
    categorical_cols = train_features_df.select_dtypes(include=['object']).columns.tolist()
    train_enc, test_enc, _ = encode_categorical_features(
        train_features_df, test_features_df, categorical_cols
    )
    
    feature_names = train_enc.columns.tolist()
    scaler_params = fit_minmax_scaler(train_enc)
    
    X_train_scaled = transform_minmax_scaler(train_enc, scaler_params)
    X_test_scaled = transform_minmax_scaler(test_enc, scaler_params)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'feature_names': feature_names,
        'scaler_params': scaler_params,
        'train_df_raw': train_df,
        'test_df_raw': test_df
    }

