# Purpose: Loads the prepared data, trains Random Forest and Isolation Forest

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def load_data():
    return joblib.load("Results/preparedData_data.joblib")

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model
    
def test_random_forest(model, x_test, y_test):
    predictions = model.predict(x_test)
    
    precision = precision_score(y_test, predictions, zero_division=0)
    
    recall = recall_score(y_test, predictions, zero_division=0)
    
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, predictions).ravel()
    
    return {
    "Model": "Random Forest",
    "Precision": precision,
    "Recall": recall,
    "f1_score": f1,
    "true_positive": true_positive,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "true_negative": true_negative,
    }
    
def train_isolation_forest(x_train):
    model = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
    model.fit(x_train)
    return model
    
def test_isolation_forest(model, x_test, y_test):
    predictions = model.predict(x_test)
    predictions = (predictions == -1).astype(int)
    
    precision = precision_score(y_test, predictions, zero_division=0)
    
    recall = recall_score(y_test, predictions, zero_division=0)
    
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, predictions).ravel()
    
    return {
    "Model": "Isolation Forest",
    "Precision": precision,
    "Recall": recall,
    "f1_score": f1,
    "true_positive": true_positive,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "true_negative": true_negative,
    }

def main(): 
    data = load_data()
    
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    
    print("Training Random Forest")
    random_forest = train_random_forest(x_train, y_train)
    random_forest_results = test_random_forest(random_forest, x_test, y_test)
    print("Random Forest Results: ", round(random_forest_results["f1_score"], 4))
    
    print("Training Isolation Forest")
    isolation_forest = train_isolation_forest(x_train)
    isolation_forest_results = test_isolation_forest(isolation_forest, x_test, y_test)
    print("Isolation Forest Results: ", round(isolation_forest_results["f1_score"], 4))
    
    results = pd.DataFrame(
        [
            random_forest_results,
            isolation_forest_results,
        ]
    )
    
    results.to_csv(
        "Results/model_comparison.csv",
        index=False,
    )

    joblib.dump(random_forest, "Results/random_forest_model.joblib")
    
    joblib.dump(isolation_forest, "Results/isolation_forest_model.joblib")
    
    print("\n Model Comparison: ")
    print(results)
    
if __name__ == "__main__":
    main()

