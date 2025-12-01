# ml_models.py (FIXED: Categorical column handling in Heart Disease dataset)

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer # Missing values fill karne ke liye
from sklearn.preprocessing import LabelEncoder # Strings ko numbers mein badalna

# --- File Paths ---
DIABETES_PATH = os.path.join('datasets', 'diabetes.csv')
HEART_PATH = os.path.join('datasets', 'heart_disease.csv')
SLEEP_STRESS_PATH = os.path.join('datasets', 'sleep_stress.csv')
OBESITY_PATH = os.path.join('datasets', 'obesity_risk.csv')

def load_data(file_path):
    """CSV file ko load karta hai."""
    try:
        # 'low_memory=False' use kiya taa'ke data types theek se detect hon
        return pd.read_csv(file_path, low_memory=False) 
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def train_and_save_models():
    """
    R2.2: Data clean karna.
    R3.1: Models ko train karna aur save (pickle) karna.
    """
    
    # ----------------------------------------------------
    # A. Heart Disease Prediction Model (FIXED)
    # ----------------------------------------------------
    print("\n--- A. Heart Disease Model Preparation ---")
    heart_df = load_data(HEART_PATH)
    if heart_df is None: return

    # R2.2 Data Cleaning:
    # 1. Target variable 'num' ko binary banana
    heart_df['target'] = heart_df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Categorical (String) columns ko numbers mein badalna (Encoding)
    # cp column ko number mein badalna zaroori hai
    encoder = LabelEncoder()
    heart_df['cp'] = encoder.fit_transform(heart_df['cp'])
    heart_df['sex'] = encoder.fit_transform(heart_df['sex'])
    
    # 'dataset' jaisi columns jinhe humein nahi chahiye unko drop kar sakte hain.
    # Hum sirf un columns ko rakhenge jin par SimpleImputer chalega (numeric)
    
    # 3. Missing values ko Impute karna (Median fill)
    # Hum sirf numeric columns par Imputer chalayenge
    cols_to_impute = ['trestbps', 'chol', 'thalch', 'oldpeak']
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    heart_df[cols_to_impute] = imputer.fit_transform(heart_df[cols_to_impute])
    
    # 'ca' aur 'thal' mein bohot zyada missing values hone ki wajah se, hum unhein features se nikal denge.
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalch', 'oldpeak']
    X_heart = heart_df[features]
    y_heart = heart_df['target']
    
    # Model Training
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    model_heart = LogisticRegression(max_iter=5000) # Max iterations increase kiya hai
    model_heart.fit(X_train_h, y_train_h)
    
    print(f"Heart Model Accuracy: {accuracy_score(y_test_h, model_heart.predict(X_test_h)):.2f}")
    
    # Model ko save karna
    with open('model_heart.pkl', 'wb') as file:
        pickle.dump(model_heart, file)
    print("Heart Disease model saved as model_heart.pkl")

    # ----------------------------------------------------
    # B. Diabetes Prediction Model 
    # ----------------------------------------------------
    print("\n--- B. Diabetes Model Preparation ---")
    diabetes_df = load_data(DIABETES_PATH)
    if diabetes_df is None: return

    # Is dataset mein 0 values ko missing samajhte hain
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    diabetes_df[cols_to_replace] = diabetes_df[cols_to_replace].replace(0, np.nan)
    
    # Missing values (0s ko nan kiya tha) ko median se fill karna
    diabetes_df[cols_to_replace] = SimpleImputer(strategy='median').fit_transform(diabetes_df[cols_to_replace])
    
    # Feature aur Target select karna
    features_d = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_diabetes = diabetes_df[features_d]
    y_diabetes = diabetes_df['Outcome']
    
    # Model Training
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
    model_diabetes = LogisticRegression(max_iter=5000) # Max iterations increase kiya hai
    model_diabetes.fit(X_train_d, y_train_d)
    
    print(f"Diabetes Model Accuracy: {accuracy_score(y_test_d, model_diabetes.predict(X_test_d)):.2f}")
    
    # Model ko save karna
    with open('model_diabetes.pkl', 'wb') as file:
        pickle.dump(model_diabetes, file)
    print("Diabetes model saved as model_diabetes.pkl")

if __name__ == '__main__':
    train_and_save_models()