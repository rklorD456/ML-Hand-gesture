# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow

# Import your helper functions
from helper import preprocess_landmarks, train_and_log_model

def run_training():
    # 1. Setup
    DATA_PATH = 'dataset/hand_landmarks_data.csv'
    mlflow.set_experiment("Hand_Gesture_Classification")

    # 2. Load and Clean Data
    df = pd.read_csv(DATA_PATH)
    df_clean = df.drop_duplicates()
    
    X = df_clean.drop('label', axis=1).values
    y = df_clean['label'].values

    # 3. Preprocess
    X_processed = np.apply_along_axis(preprocess_landmarks, 1, X)
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 4. Split
    X_train, X_test_validate, y_train, y_test_validate = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_test_validate, y_test_validate, test_size=0.5, random_state=42, stratify=y_test_validate
    )

    # 5. Define and Train Models
    
    # train model: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    train_and_log_model(
        rf_model, "Random_Forest", 
        X_train, y_train, X_validation, y_validation, 
        rf_params, dataset_path=DATA_PATH
    )

    # train model: XGBoost
    xgb_params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5}
    xgb_model = XGBClassifier(**xgb_params)
    train_and_log_model(
        xgb_model, "XGBoost_Final", 
        X_train, y_train, X_validation, y_validation, 
        xgb_params, dataset_path=DATA_PATH
    )

if __name__ == "__main__":
    run_training()