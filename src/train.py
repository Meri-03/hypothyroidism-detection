import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from preprocess import run_preprocessing_pipeline

def train_all_models(X_train, y_train):

    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name.replace('_', ' ').upper()} ---")

        # 5-fold cross validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='f1', n_jobs=-1
        )
        print(f"Cross-val F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save model
        joblib.dump(model, f'models/{name}.pkl')
        print(f"Saved → models/{name}.pkl ")

    return trained_models


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
        'data/raw/hypothyroid.data'
    )

    
    print("        STARTING MODEL TRAINING")

    trained_models = train_all_models(X_train, y_train)

    print("     ALL 3 MODELS TRAINED AND SAVED")
    
    print("Files saved in models/ folder:")
    print("  - logistic_regression.pkl")
    print("  - random_forest.pkl")
    print("  - gradient_boosting.pkl")