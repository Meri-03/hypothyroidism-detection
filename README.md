# Machine Learning-Based CDSS for Hypothyroidism Prediction

**Course:** Artificial Intelligence | INSC3112  
**Institution:** University of Gondar, College of Informatics  
**Department:** Information Science  
**Submitted to:** Mr. Melsew  

## Group Members
| Name | Student ID |
|------|-----------|
| Addis Alem Brhanu | 01078/16 |
| Aboker Semagn | 22888/16 |
| Samrawit Getahun | 02888/16 |
| Seble Getnet | 00704/16 |
| Meron Ashenafi | 01679/16 |
| Victoria Tsegalign | 02865/16 |

---

## Project Overview
A machine learning-based Clinical Decision Support System (CDSS) 
for early prediction of hypothyroidism in Ethiopian healthcare settings.

## Dataset
- Source: UCI Hypothyroid Dataset
- Patients: 3,163
- Features: 20 (after preprocessing)
- Classes: Normal (3,012) | Hypothyroid (151)

## Models Trained
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 94.94% | 0.9424 |
| Random Forest | 94.94% | 0.9775 |
| Gradient Boosting  | 96.05% | 0.9786 |

## Project Structure
hypothyroidism/
├── data/
│   ├── raw/hypothyroid.data        # Original dataset
│   └── processed/cleaned_data.csv  # After preprocessing
├── src/
│   ├── load_data.py                # Data loading
│   ├── preprocess.py               # Cleaning pipeline
│   ├── train.py                    # Model training
│   └── evaluate.py                 # Metrics + charts
├── app/
│   └── streamlit_app.py            # Web UI
├── models/                         # Saved .pkl files
├── reports/                        # Charts + final report
├── notebooks/                      # EDA notebooks
└── requirements.txt
## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run preprocessing
```bash
python src/preprocess.py
```

### 3. Train models
```bash
python src/train.py
```

### 4. Evaluate models
```bash
python src/evaluate.py
```

### 5. Launch the web app
```bash
streamlit run app/streamlit_app.py
```
Then follow the link.

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- streamlit
- imbalanced-learn
- matplotlib
- seaborn
- joblib

  
----------------------Gracias-----------------------------------
