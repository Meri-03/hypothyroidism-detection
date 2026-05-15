import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_data

# ─── Step 1: Clean target ──────────────────────────────────────────────────────
def clean_target(df):
    df = df.copy()
    df['target'] = df['target'].astype(str).str.strip().str.rstrip('.')
    df['target'] = df['target'].apply(
        lambda x: 1 if 'hypothyroid' in x.lower() else 0
    )
    print(f"Binary target — 0 (normal): {(df['target']==0).sum()}, 1 (hypothyroid): {(df['target']==1).sum()}")
    return df

# ─── Step 2: Drop useless columns ─────────────────────────────────────────────
def drop_columns(df):
    cols_to_drop = [
        'TSH_measured', 'T3_measured', 'TT4_measured',
        'T4U_measured', 'FTI_measured', 'TBG_measured',
        'TBG', 'referral_source',
        'FTI'   # ← added: entire column is NaN in this dataset
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Columns remaining after drop: {len(df.columns)}")
    return df

# ─── Step 3: Handle missing values ────────────────────────────────────────────
def handle_missing(df):
    df = df.copy()

    # Separate feature columns (exclude target)
    feature_cols = [c for c in df.columns if c != 'target']

    for col in feature_cols:
        if df[col].dtype == 'object':
            # Categorical: fill with most frequent value
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
        else:
            # Numerical: fill with median
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)

    remaining = df[feature_cols].isnull().sum().sum()
    print(f"Missing values after cleaning: {remaining}")

    if remaining > 0:
        print("Columns still with NaN:")
        print(df[feature_cols].isnull().sum()[df[feature_cols].isnull().sum() > 0])

    return df

# ─── Step 4: Encode categorical columns ───────────────────────────────────────
def encode_categoricals(df):
    df = df.copy()

    binary_map = {
        't': 1, 'f': 0,
        'y': 1, 'n': 0,
        'M': 1, 'F': 0,
        'm': 1, 'f': 0
    }

    feature_cols = [c for c in df.columns if c != 'target']

    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map(binary_map)
            # If any value didn't map, fill with 0
            df[col] = df[col].fillna(0).astype(int)

    return df

# ─── Step 5: Scale features ────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

# ─── Step 6: SMOTE ────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    # Final NaN check before SMOTE
    if np.isnan(X_train).any():
        nan_count = np.isnan(X_train).sum()
        raise ValueError(f"NaN values still present before SMOTE: {nan_count} NaNs found.")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    print(f"After  SMOTE — 0: {(y_res==0).sum()}, 1: {(y_res==1).sum()}")

    return X_res, y_res

# ─── MASTER PIPELINE ──────────────────────────────────────────────────────────
def run_preprocessing_pipeline(filepath):
    # 1. Load
    df = load_data(filepath)

    # 2. Clean target
    df = clean_target(df)

    # 3. Drop useless columns
    df = drop_columns(df)

    # 4. Fix missing values
    df = handle_missing(df)

    # 5. Encode categoricals
    df = encode_categoricals(df)

    # 6. Verify no NaN remains in features
    X = df.drop('target', axis=1)
    y = df['target']

    assert X.isnull().sum().sum() == 0, "ERROR: NaN still in X after encoding!"
    assert y.isnull().sum() == 0,       "ERROR: NaN still in y!"

    print(f"\nFeature columns ({len(X.columns)}): {list(X.columns)}")

    # Save feature names for Streamlit
    joblib.dump(list(X.columns), 'models/feature_names.pkl')

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Scale
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 9. SMOTE
    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)

    # Save test data for evaluation later
    joblib.dump((X_test_scaled, y_test), 'models/test_data.pkl')
    # Save cleaned data to processed folder
    df.to_csv('data/processed/cleaned_data.csv', index=False)

    print(f"\n PIPELINE COMPLETE")
    print(f"Train size: {X_train_final.shape}")
    print(f"Test size:  {X_test_scaled.shape}")

    return X_train_final, X_test_scaled, y_train_final, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
        'data/raw/hypothyroid.data'
    )