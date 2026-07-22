# Purpose: Loads the prepared data, trains Random Forest and Isolation Forest

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import time

def load_data():
    return joblib.load("Results/prepare_data.joblib")

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model
    
def test_random_forest(model, x_test, y_test):
    
    start_time = time.perf_counter()
    
    predictions = model.predict(x_test)
    
    total_time = time.perf_counter() - start_time
    latency = (total_time * 1000) / len(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    
    precision = precision_score(y_test, predictions, zero_division=0)
    
    recall = recall_score(y_test, predictions, zero_division=0)
    
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    roc_auc = roc_auc_score( y_test, probabilities)
    
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, predictions).ravel()
    
    return {
    "Model": "Random Forest",
    "Precision": precision,
    "Recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "latency": latency,
    "true_positive": true_positive,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "true_negative": true_negative,
    }
    
def per_class_detection_rate(predictions, attack_types):
    results = pd.DataFrame({"attack_type": attack_types.values, "predicted": predictions})
    
    counts = results.groupby("attack_type").size()
    rates = results.groupby("attack_type")["predicted"].mean()
    
    completion = pd.DataFrame({"count": counts, "detection_rate": rates})
    
    return completion.reset_index()
    
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
    
    #adding cross-validation
    average_f1 = cross_val_score(random_forest, x_train, y_train, cv=5, scoring="f1").mean()
    random_forest_results["Average_f1"] = average_f1
    print("Average F1 Score: ", average_f1)
    
    # per-class detection rate
    attack_types = data["attack_type_test"]
    predictions = random_forest.predict(x_test)
    
    breakdown = per_class_detection_rate(predictions, attack_types)
    breakdown.to_csv("Results/per_class_detection.csv", index=False)
    
    print("\nDetection results: ")
    print(breakdown)
    
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

