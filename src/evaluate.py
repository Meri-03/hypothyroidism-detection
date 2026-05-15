import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve
)

# ─── Load saved test data and models ──────────────────────────────────────────
def load_artifacts():
    X_test, y_test       = joblib.load('models/test_data.pkl')
    feature_names        = joblib.load('models/feature_names.pkl')

    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest':       joblib.load('models/random_forest.pkl'),
        'Gradient Boosting':   joblib.load('models/gradient_boosting.pkl'),
    }
    return models, X_test, y_test, feature_names

# ─── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred),    4),
        'F1-Score':  round(f1_score(y_test, y_pred),        4),
        'AUC-ROC':   round(roc_auc_score(y_test, y_prob),   4),
    }

    print(f"\n{'='*55}")
    print(f"  {name.upper()}")
    print(f"{'='*55}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Hypothyroid']
    ))

    return metrics, y_pred, y_prob

# ─── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Hypothyroid'],
        yticklabels=['Normal', 'Hypothyroid']
    )
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    filename = f"reports/confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved → {filename}")

# ─── ROC curves for all models ────────────────────────────────────────────────
def plot_roc_curves(roc_data):
    plt.figure(figsize=(7, 5))

    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve — Model Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/roc_curves.png', dpi=150)
    plt.show()
    print("Saved → reports/roc_curves.png")

# ─── Feature importance (Random Forest) ───────────────────────────────────────
def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    plt.barh(
        range(top_n),
        importances[indices],
        color='steelblue', alpha=0.85
    )
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=150)
    plt.show()
    print("Saved → reports/feature_importance.png")

# ─── Model comparison bar chart ───────────────────────────────────────────────
def plot_comparison_bar(results_df):
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_plot   = results_df.set_index('Model')[metrics]

    ax = df_plot.plot(kind='bar', figsize=(10, 5), width=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0.8, 1.02)
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/model_comparison_chart.png', dpi=150)
    plt.show()
    print("Saved → reports/model_comparison_chart.png")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Make sure reports folder exists
    os.makedirs('reports', exist_ok=True)

    models, X_test, y_test, feature_names = load_artifacts()

    all_metrics = []
    roc_data    = {}

    for name, model in models.items():
        metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics['AUC-ROC'])

        plot_confusion_matrix(name, y_test, y_pred)

    # Summary table
    results_df = pd.DataFrame(all_metrics)
    print("\n" + "="*55)
    print("         FINAL MODEL COMPARISON TABLE")
    print("="*55)
    print(results_df.to_string(index=False))
    results_df.to_csv('reports/model_comparison.csv', index=False)
    print("\nSaved → reports/model_comparison.csv")

    # Charts
    plot_roc_curves(roc_data)
    plot_feature_importance(
        models['Random Forest'], feature_names
    )
    plot_comparison_bar(results_df)

    print("\n EVALUATION COMPLETE — all charts saved in reports/")