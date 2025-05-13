import os
from datetime import datetime, timedelta, time
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

"""
ml_model.py — Predicts bullish and bearish trade setups using engineered financial features.
Uses Random Forest and XGBoost with GridSearchCV and calibrated outputs.
"""

ENTRY_THRESHOLD = 0.7
EXIT_THRESHOLD = 0.05
EXIT_WINDOW = 5

BASE_DROP_COLS = [
    ...
]

param_grid = {
...
}
base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(
    base_model, 
    param_grid, 
    scoring='f1', 
    cv=3, 
    n_jobs=-1, 
    verbose=0
)
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)
xgb_param_grid = {
    ...
}
xgb_grid_search = GridSearchCV(
    xgb_model,
    xgb_param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=0
)

def load_and_engineer_data():
    ...

def add_null_flags(df, drop_cols):
    null_flags = {
        f"{col}_was_null": df[col].isnull().astype(int)
        for col in df.columns
        if df[col].isnull().any() and col not in drop_cols
    }
    return pd.concat([df, pd.DataFrame(null_flags, index=df.index)], axis=1)

def debug_entry_filters(df, test_df, direction):
    ...

def check_bullish(df, drop_cols):
    ...

def check_bearish(df, drop_cols):
    ...

def train_model(df, drop_cols, target_name):
    ...
    
def predict_probabilities(calibrated_model, X_test, y_test, target_name):
    ...
    
def evaluate_model(target_name, calibrated_model, X_test, y_test):
    ...
    
def check_prediction_accuracy(df_combined, df):
    ...

def prepare_results_df(df, target_name, y_test, probs, preds, test_df):
    ...
    
def log_model_performance(accuracy, precision, recall, f1, X_train, X_test, y_train, y_test, drop_cols, importances, best_threshold, target_name):
    ...
    
def main():
    df, drop_cols = load_and_engineer_data()
    check_bullish(df, drop_cols)
    check_bearish(df, drop_cols)
    
if __name__ == "__main__":
    main()