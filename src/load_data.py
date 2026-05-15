import pandas as pd
import numpy as np

COLUMN_NAMES = [
    'target',                    # FIRST column in this file
    'age', 'sex',
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH',
    'T3_measured', 'T3',
    'TT4_measured', 'TT4',
    'T4U_measured', 'T4U',
    'FTI_measured', 'FTI',
    'TBG_measured', 'TBG',
    'referral_source'
]

def load_data(filepath):
    df = pd.read_csv(filepath, names=COLUMN_NAMES, na_values='?')

    print(f"\n--- SUCCESS: DATA LOADED ---")
    print(f"Total Patients: {df.shape[0]}")
    print(f"Total Features: {df.shape[1]}")
    print(f"\nRaw target values:")
    print(df['target'].value_counts(dropna=False))

    return df