import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.datasets import make_classification

def load_and_engineer_data():
    print("--- Generating Simulated Fraud Data ---")
    X_raw, y_raw = make_classification(
        n_samples=50000, n_features=15, n_informative=10, 
        n_redundant=5, weights=[0.995, 0.005], random_state=42
    )
    
    cols = [f'TransactionAmt', 'card1', 'card2', 'addr1', 'dist1'] + [f'C{i}' for i in range(1, 11)]
    df = pd.DataFrame(X_raw, columns=cols)
    df['isFraud'] = y_raw
    df['TransactionDT'] = np.arange(len(df)) * 10 
    
    print("--- Feature Engineering ---")
    df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24
    
    card1_mean_amt = df.groupby('card1')['TransactionAmt'].transform('mean')
    card1_std_amt = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
    
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / (card1_mean_amt + 1e-5)
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / (card1_std_amt + 1e-5)
    df['time_delta_card1'] = df.groupby('card1')['TransactionDT'].diff().fillna(-1)
    
    X = df.drop(columns=['isFraud', 'TransactionDT'])
    y = df['isFraud']
    return X, y

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0),
        'tree_method': 'hist',
        'random_state': 42,
        'eval_metric': 'aucpr'
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pr_aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
        pr_aucs.append(score)
        
    return np.mean(pr_aucs)

if __name__ == "__main__":
    X, y = load_and_engineer_data()
    
    print("--- Starting Hyperparameter Optimization ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=5)
    
    print(f"Best PR-AUC Score: {study.best_value:.4f}")
    
    print("--- Training Final Production Model ---")
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y)
    
    joblib.dump(final_model, 'fraud_model.joblib')
    joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
    print("Artifacts successfully created locally!")
