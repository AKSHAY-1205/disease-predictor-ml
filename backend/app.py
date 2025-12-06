from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load

warnings.filterwarnings("ignore")

# Try to import RandomOverSampler; fall back if not installed
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_assets")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

class RandomForestDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_trained = False
        self.metrics = {}

    def _dataset(self):
        url = "Disease_symptom_and_patient_profile_dataset.csv"
        return pd.read_csv(url)

    def _preprocess(self, df: pd.DataFrame):
        # Map categorical to numeric
        symptom_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
        for c in symptom_cols:
            df[c] = df[c].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
        df["Blood Pressure"] = df["Blood Pressure"].map({"High": 2, "Normal": 1, "Low": 0}).fillna(1).astype(int)
        df["Cholesterol Level"] = df["Cholesterol Level"].map({"High": 2, "Normal": 1, "Low": 0}).fillna(1).astype(int)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30).clip(0, 120)

        # Extra engineered features
        df["Risk_Score"] = (df["Age"] / 100.0) + (df["Blood Pressure"] * 0.3) + (df["Cholesterol Level"] * 0.2)
        df["Symptom_Count"] = df[symptom_cols].sum(axis=1)
        df["Age_Risk"] = (df["Age"] > 60).astype(int)

        # Filter rare classes (min 5 samples)
        counts = df["Disease"].value_counts()
        df = df[df["Disease"].isin(counts[counts >= 5].index)].copy()

        features = symptom_cols + [
            "Age",
            "Gender",
            "Blood Pressure",
            "Cholesterol Level",
            "Risk_Score",
            "Symptom_Count",
            "Age_Risk",
        ]

        X = df[features].astype(float)
        y = self.label_encoder.fit_transform(df["Disease"])

        # Class balance
        if HAS_IMB:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)

        self.feature_columns = features
        return X, y

    def train(self, save=True):
        try:
            df = self._dataset()
        except Exception as e:
            print(f"[model] Failed to load dataset: {e}")
            return False

        X, y = self._preprocess(df)

        # Scale
        X = self.scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Random Forest optimized for speed/accuracy
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted")
        rec = recall_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")

        print(f"[model] RF accuracy: {acc:.4f}")
        print(f"[model] RF prec: {prec:.4f}")
        print(f"[model] RF rec: {rec:.4f}")
        print(f"[model] RF f1: {f1:.4f}")

        self.metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        self.model_trained = True

        if save:
            try:
                dump(self.model, MODEL_PATH)
                dump(self.scaler, SCALER_PATH)
                dump(self.label_encoder, ENCODER_PATH)
                with open(FEATURES_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.feature_columns, f)
                with open(METRICS_PATH, "w") as f:
                    json.dump(self.metrics, f)
                print("[model] Saved model assets & metrics.")
            except Exception as e:
                print(f"[model] Failed to save model: {e}")

        return True

    def load(self):
        try:
            self.model = load(MODEL_PATH)
            self.scaler = load(SCALER_PATH)
            self.label_encoder = load(ENCODER_PATH)
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)
                print(f"[model] RF accuracy: {self.metrics['accuracy']:.4f}")
                print(f"[model] RF prec: {self.metrics['precision']:.4f}")
                print(f"[model] RF rec: {self.metrics['recall']:.4f}")
                print(f"[model] RF f1: {self.metrics['f1']:.4f}")

            self.model_trained = True
            print("[model] Loaded model assets from disk.")
            return True
        except Exception as e:
            print(f"[model] No cached model to load: {e}")
            return False

    def _vectorize(self, payload: dict):
        fever = int(payload.get("fever", 0))
        cough = int(payload.get("cough", 0))
        fatigue = int(payload.get("fatigue", 0))
        db = int(payload.get("difficulty_breathing", 0))
        age = float(payload.get("age", 30))
        gender = int(payload.get("gender", 0))
        bp = int(payload.get("blood_pressure", 1))
        chol = int(payload.get("cholesterol_level", 1))

        age = float(np.clip(age, 0, 120))
        fever = int(np.clip(fever, 0, 1))
        cough = int(np.clip(cough, 0, 1))
        fatigue = int(np.clip(fatigue, 0, 1))
        db = int(np.clip(db, 0, 1))
        bp = int(np.clip(bp, 0, 2))
        chol = int(np.clip(chol, 0, 2))
        gender = 1 if gender == 1 else 0

        base = [fever, cough, fatigue, db, age, gender, bp, chol]
        risk_score = (age / 100.0) + (bp * 0.3) + (chol * 0.2)
        symptom_count = fever + cough + fatigue + db
        age_risk = 1 if age > 60 else 0

        features = base + [risk_score, symptom_count, age_risk]
        return np.array(features, dtype=float).reshape(1, -1)

    def predict(self, payload: dict):
        if not self.model_trained:
            return {"error": "Model not trained"}
        vec = self._vectorize(payload)
        vec = self.scaler.transform(vec)
        y = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0].max()
        disease = self.label_encoder.inverse_transform([y])[0]
        risk = "High" if proba > 0.8 else ("Medium" if proba > 0.6 else "Low")
        return {
            "predicted_disease": disease,
            "confidence": float(proba),
            "risk_assessment": risk,
        }

predictor = RandomForestDiseasePredictor()

def ensure_model():
    if predictor.model_trained:
        return True
    if predictor.load():
        return True
    print("[model] Training fresh model...")
    return predictor.train(save=True)

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "model": "RandomForest",
        "endpoints": ["/predict", "/health", "/model-info"],
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "healthy": True,
        "model_trained": predictor.model_trained,
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    if predictor.model_trained and predictor.feature_columns:
        return jsonify({
            "model_type": "RandomForestClassifier",
            "features": predictor.feature_columns,
            "classes": list(predictor.label_encoder.classes_),
            "metrics": predictor.metrics if predictor.metrics else "Unavailable"
        })
    else:
        return jsonify({"error": "Model not trained"}), 503

@app.route("/predict", methods=["POST"])
def predict():
    try:
        ok = ensure_model()
        if not ok:
            return jsonify({"error": "Model unavailable"}), 503

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        required = [
            "fever",
            "cough",
            "fatigue",
            "difficulty_breathing",
            "age",
            "gender",
            "blood_pressure",
            "cholesterol_level",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        result = predictor.predict(data)
        if "error" in result:
            return jsonify(result), 500

        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
