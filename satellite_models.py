"""
Satellite Active/Inactive classification: five-model comparison.
Run: pip install pandas numpy scikit-learn openpyxl
     python satellite_models.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("SpaceDatasetCleaned.xlsx")
df["Launch_Month"] = df["LAUNCH_DATE"].dt.month
df["Launch_DayOfYear"] = df["LAUNCH_DATE"].dt.dayofyear

# Exclude direct target leakage and identifiers.
df = df.drop(columns=["Satellite", "OBJECT_TYPE", "OPS_STATUS_CODE", "DECAY_DATE", "LAUNCH_DATE"])
X = df.drop(columns=["Active_Status"])
y = df["Active_Status"].map({"Inactive": 0, "Active": 1})

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(include=[np.number]).columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical)
])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=42, cache_size=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)
}

def run_evaluation(X_train, X_test, y_train, y_test, split_name):
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        start = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - start
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Split": split_name, "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "Fit time (sec)": fit_time, "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
    return rows

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
results = run_evaluation(X_train, X_test, y_train, y_test, "Random 80/20")

# Optional time-based stress test.
launch_year = pd.read_excel("SpaceDatasetCleaned.xlsx")["Launch Year"]
train_mask = launch_year <= 2018
results += run_evaluation(X.loc[train_mask], X.loc[~train_mask],
                          y.loc[train_mask], y.loc[~train_mask],
                          "Temporal <=2018 vs 2019+")

out = pd.DataFrame(results).sort_values(["Split", "F1-score"], ascending=[True, False])
print(out.to_string(index=False))
out.to_excel("Satellite_Model_Metrics.xlsx", index=False)
