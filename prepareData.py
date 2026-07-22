# Purpose: Loads the raw data, cleans it, and prepares train/test splits ready fro model training
import glob
import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.prepareData import StandardScaler

def load_data():
    files = glob.glob("data/raw/*.csv")

    if not files:
        raise FileNotFoundError("No files found")

    dataFrames = []

    for file in files: 
        print(f"Reading {file}")
        dataFrame = pd.read_csv(file, low_memory=False)
        dataFrames.append(dataFrame)
        
    data = pd.concat(dataFrames, ignore_index=True)
    data.columns = data.columns.str.strip()
    
    return data
    
def clean_data(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    
    return data
    
def prepare_data(data):
    if "Label" not in data.columns:
        raise ValueError("No label column")
        
    data["is_attack"] = ( data["Label"].astype(str).str.strip().str.upper() 
    != "Benign").astype(int)
    
    x = data.drop(columns=["Label", "is_attack"])
    y = data["is_attack"]
    
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.dropna(axis=1, how="all")
    
    valid_rows = x.notna().all(axis=1)
    x = x.loc[valid_rows]
    y = y.loc[valid_rows]
    
    x_train, x_test, y_train, y_test = train_test_split (
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    
    #scaling
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    smote = SMOTE(random_state=42)
    
    x_train, y_train = smote.fit_resample(x_train, y_train)
    
    return {
    "x_train": x_train,
    "y_train": y_train,
    "x_test": x_test,
    "y_test": y_test,
    "Scaler": scaler,
    "Names": list(x.columns),
    }

def main():
    os.makedirs("Results", exist_ok=True)
    
    data = load_data()
    print(f"Loaded {len(data)} Rows")
    
    data = clean_data(data)
    print(f"Loaded {len(data)} Rows")
    
    processed_data = prepare_data(data)
    
    output_file = "Results/prepareData_data.joblib"
    joblib.dump(processed_data, output_file)
    
    print(f"Training: {len(processed_data['x_train'])}")
    print(f"Testing: {len(processed_data['x_test'])}")
    
if __name__ == "__main__":
    main()

