import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("improved_realistic_disease_dataset_50000.csv")

# -----------------------------
# Encode categorical columns
# -----------------------------
categorical_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", 
                    "Gender", "Blood Pressure", "Cholesterol Level"]

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Features & target
X = df_encoded.drop("Disease", axis=1)
y = df_encoded["Disease"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Handle class imbalance with SMOTE
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -----------------------------
# Hyperparameter tuning for Random Forest
# -----------------------------
rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf_search = RandomizedSearchCV(
    rf, param_grid, n_iter=30, cv=5, scoring="accuracy", n_jobs=-1, random_state=42
)
rf_search.fit(X_train_res, y_train_res)

best_rf = rf_search.best_estimator_

# -----------------------------
# Train the best Random Forest
# -----------------------------
best_rf.fit(X_train_res, y_train_res)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Random Forest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model and encoders
# # -----------------------------
# joblib.dump(best_rf, "best_disease_model.pkl")
# joblib.dump(X.columns, "feature_columns.pkl")
print("\n✅ Random Forest model and feature columns saved successfully!")
