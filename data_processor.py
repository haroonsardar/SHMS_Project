# data_processor.py (Finalized Paths Ke Saath Updated Code)

import pandas as pd
import os

# 1. Datasets ka Path (Finalized File Names)
# Yaqeen karein ki yeh files aapke 'datasets' folder mein maujood hain aur inhi naam se hain.
DIABETES_PATH = os.path.join('datasets', 'diabetes.csv')
HEART_PATH = os.path.join('datasets', 'heart_disease.csv') 
SLEEP_STRESS_PATH = os.path.join('datasets', 'sleep_stress.csv')
OBESITY_PATH = os.path.join('datasets', 'obesity_risk.csv')

def load_and_inspect_data(file_path, file_name):
    """
    Data ko load karta hai aur uski pehli 5 rows aur missing values ka jaiza leta hai.
    """
    try:
        # File ko Pandas DataFrame mein load karna
        df = pd.read_csv(file_path)
        
        print(f"\n--- {file_name} Loaded Successfully ---")
        
        # Pehli 5 rows dikhana
        print("Pehli 5 rows (df.head()):")
        print(df.head())
        
        # Missing values (NaN) ko count karna
        print("\nMissing Values Count (df.isnull().sum()):")
        print(df.isnull().sum())
        
        return df
    
    except FileNotFoundError:
        print(f"\nERROR: File not found at {file_path}. Please check the file name and path.")
        return None
    except Exception as e:
        print(f"\nERROR loading {file_name}: {e}")
        return None

# Main execution
if __name__ == '__main__':
    print("--- Phase 1.4: Data Loading Aur Initial Inspection Shuru Ho Rahi Hai ---")
    
    # Har dataset ko load aur inspect karna
    diabetes_df = load_and_inspect_data(DIABETES_PATH, "Diabetes Dataset")
    heart_df = load_and_inspect_data(HEART_PATH, "Heart Disease Dataset")
    sleep_stress_df = load_and_inspect_data(SLEEP_STRESS_PATH, "Sleep & Stress Dataset")
    obesity_df = load_and_inspect_data(OBESITY_PATH, "Lifestyle & Obesity Dataset")