# =========================================================
# Disease Prediction Model (Leakage-Free, Colab Ready)
# =========================================================

# If running in a fresh Colab, uncomment:
# !pip install xgboost scikit-learn pandas numpy matplotlib seaborn plotly torch torch-geometric imbalanced-learn torch-scatter torch-sparse -q

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ---- GNN bits ----
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph


# =========================================================
# Core Predictor Class
# =========================================================
class DiseasePredictor:
    def __init__(self):
        # Trained models & metadata
        self.models = {}
        self.feature_names = []
        self.disease_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Holdouts for evaluation/plots
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_names = None
        self.y_test_names = None

        # Original preprocessed dataframe
        self.data = None

    # ---------------------------
    # Data Loading & Preprocessing
    # ---------------------------
    def load_and_preprocess_data(self, csv_path=None, df=None):
        """
        Load CSV or provided DataFrame and apply:
        - category encoding
        - missing value handling
        - light feature engineering
        """
        if df is not None:
            self.data = df.copy()
        else:
            if csv_path is None:
                raise ValueError("Provide either csv_path or df.")
            self.data = pd.read_csv(csv_path)

        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")

        # Quick look at class distribution
        if 'Disease' in self.data.columns:
            print("\nTop disease counts:")
            print(self.data['Disease'].value_counts().head(10))

        self._encode_categorical_features()
        self._handle_missing_values()
        self._feature_engineering()
        return self.data

    def _encode_categorical_features(self):
        # Binary symptom columns expected as Yes/No
        binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
        for col in binary_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].map({'Yes': 1, 'No': 0})

        # Gender to 0/1
        if 'Gender' in self.data.columns:
            self.data['Gender'] = self.data['Gender'].map({'Male': 1, 'Female': 0})

        # Ordinal encodings
        bp_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
        chol_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
        if 'Blood Pressure' in self.data.columns:
            self.data['Blood Pressure'] = self.data['Blood Pressure'].map(bp_mapping)
        if 'Cholesterol Level' in self.data.columns:
            self.data['Cholesterol Level'] = self.data['Cholesterol Level'].map(chol_mapping)

        # Outcome variable to 0/1 if present
        if 'Outcome Variable' in self.data.columns:
            self.data['Outcome Variable'] = self.data['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    def _handle_missing_values(self):
        # Fill numerics with median
        for col in ['Age', 'Blood Pressure', 'Cholesterol Level']:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())

        # Fill binaries/categoricals with mode
        for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Outcome Variable']:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def _feature_engineering(self):
        # Derived features
        symptom_cols = [c for c in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'] if c in self.data.columns]
        if symptom_cols:
            self.data['Total_Symptoms'] = self.data[symptom_cols].sum(axis=1)
        else:
            self.data['Total_Symptoms'] = 0

        # Age group buckets
        if 'Age' in self.data.columns:
            self.data['Age_Group'] = pd.cut(
                self.data['Age'],
                bins=[-1, 30, 45, 60, 120],
                labels=[0, 1, 2, 3]
            ).astype(int)
        else:
            self.data['Age_Group'] = 0

        # Risk Score: avoid NaNs if cols are missing
        age = self.data['Age'] if 'Age' in self.data.columns else 0
        bp = self.data['Blood Pressure'] if 'Blood Pressure' in self.data.columns else 0
        chol = self.data['Cholesterol Level'] if 'Cholesterol Level' in self.data.columns else 0
        self.data['Risk_Score'] = (age / 100.0) + (bp * 0.3) + (chol * 0.3)

    # ---------------------------
    # Preparing splits WITHOUT leakage
    # ---------------------------
    def prepare_splits(self):
        feature_cols = [
            col for col in [
                'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
                'Total_Symptoms', 'Age_Group', 'Risk_Score'
            ] if col in self.data.columns
        ]
        self.feature_names = feature_cols

        X = self.data[feature_cols].copy()
        y_disease = self.data['Disease'].copy()

        # Encode disease labels once
        y_disease_encoded = self.disease_encoder.fit_transform(y_disease)

        # Split before scaling (to avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_disease_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded
        )

        # Fit scaler ONLY on train
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(self.scaler.transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),  columns=feature_cols, index=X_test.index)

        # Keep names for reports
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.y_train_names = self.disease_encoder.inverse_transform(y_train)
        self.y_test_names  = self.disease_encoder.inverse_transform(y_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ---------------------------
    # Imbalance handling
    # ---------------------------
    def handle_class_imbalance(self, X, y):
        smote = SMOTE(random_state=42)
        Xb, yb = smote.fit_resample(X, y)
        return Xb, yb

    # ---------------------------
    # Train Classical Models
    # ---------------------------
    def train_traditional_models(self):
        assert self.X_train is not None, "Call prepare_splits() first."

        # Apply SMOTE ONLY on training data
        X_train_bal, y_train_bal = self.handle_class_imbalance(self.X_train, self.y_train)

        # Models with reasonable regularization
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_lambda=1.5,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1,
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight=None,     # using SMOTE instead
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                ccp_alpha=0.001,       # prune a bit
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
        }

        results = {}

        # 5-fold CV on TRAIN (balanced)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            print(f"\nTraining {name} ...")
            model.fit(X_train_bal, y_train_bal)

            # Train metrics (on balanced train)
            y_train_pred = model.predict(X_train_bal)
            train_acc = accuracy_score(y_train_bal, y_train_pred)

            # Test metrics
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

            test_acc = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            # CV score on balanced train
            cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='accuracy')

            print(f"{name} -> Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | "
                  f"F1: {f1:.3f} | CV: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

            results[name] = {
                'model': model,
                'train_acc': train_acc,
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_proba
            }

        self.models.update(results)
        return results

    # ---------------------------
    # GNN Model (lightweight)
    # ---------------------------
    def create_gnn_model(self, n_neighbors=8, epochs=60, lr=0.01, weight_decay=5e-4, hidden_dim=64, dropout=0.4):
        """
        Build a sample GNN over an instance graph. Uses standardized features
        (same scaler fit on train). Uses the same train/test split masks.
        """
        class DiseaseGNN(torch.nn.Module):
            def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.4):
                super().__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.dropout = torch.nn.Dropout(dropout)
                self.classifier = torch.nn.Linear(hidden_dim, num_classes)

            def forward(self, x, edge_index, batch):
                x = F.relu(self.conv1(x, edge_index))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, edge_index))
                x = self.dropout(x)
                x = global_mean_pool(x, batch)
                x = self.classifier(x)
                return F.log_softmax(x, dim=1)

        # Build feature matrix over *all* rows (X_train + X_test) to make one graph
        X_all = pd.concat([self.X_train, self.X_test], axis=0)
        y_all = np.concatenate([self.y_train, self.y_test], axis=0)
        num_samples = X_all.shape[0]
        num_features = X_all.shape[1]
        num_classes = len(np.unique(y_all))

        # kNN graph on standardized features
        A = kneighbors_graph(X_all.values, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

        x_tensor = torch.tensor(X_all.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_all, dtype=torch.long)

        # masks
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask[:self.X_train.shape[0]] = True
        test_mask[self.X_train.shape[0]:] = True

        # Batch: single component graph (all zeros)
        batch = torch.zeros(num_samples, dtype=torch.long)

        # Model/optim/criterion
        model = DiseaseGNN(num_features, num_classes, hidden_dim=hidden_dim, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.NLLLoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(x_tensor, edge_index, batch)
            loss = criterion(out[train_mask], y_tensor[train_mask])
            loss.backward()
            optimizer.step()
            if (epoch+1) % 15 == 0:
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    train_acc = (pred[train_mask] == y_tensor[train_mask]).float().mean().item()
                print(f"GNN Epoch {epoch+1:03d} | Loss {loss:.4f} | Train Acc {train_acc:.3f}")

        model.eval()
        with torch.no_grad():
            out = model(x_tensor, edge_index, batch)
            pred = out.argmax(dim=1)
            test_acc = (pred[test_mask] == y_tensor[test_mask]).float().mean().item()
        print(f"GNN Test Accuracy: {test_acc:.3f}")

        self.models['GNN'] = {
            'model': model,
            'accuracy': test_acc,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'x': x_tensor,
            'edge_index': edge_index,
            'batch': batch
        }
        return model, test_acc

    # ---------------------------
    # Inference
    # ---------------------------
    def predict_disease(self, symptoms_dict):
        assert self.feature_names, "Train models first to establish feature order & scaler."

        # Start with zeros for all features, then fill
        fv = {name: 0 for name in self.feature_names}

        # Map external keys to internal columns
        key_map = {
            'fever': 'Fever',
            'cough': 'Cough',
            'fatigue': 'Fatigue',
            'difficulty_breathing': 'Difficulty Breathing',
            'age': 'Age',
            'gender': 'Gender',
            'blood_pressure': 'Blood Pressure',
            'cholesterol_level': 'Cholesterol Level',
        }

        for k, v in symptoms_dict.items():
            if k.lower() in key_map:
                fv[key_map[k.lower()]] = v

        # Derived
        fv['Total_Symptoms'] = sum([fv.get(c, 0) for c in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']])
        age = fv.get('Age', 0)
        fv['Age_Group'] = 0 if age <= 30 else (1 if age <= 45 else (2 if age <= 60 else 3))
        fv['Risk_Score'] = (age / 100.0) + (fv.get('Blood Pressure', 0) * 0.3) + (fv.get('Cholesterol Level', 0) * 0.3)

        # Scale using train-fitted scaler
        fv_vec = [fv[name] for name in self.feature_names]
        fv_scaled = self.scaler.transform([fv_vec])[0]

        predictions = {}
        for name, m in self.models.items():
            if name == 'GNN':
                continue
            model = m['model']
            pred_enc = model.predict([fv_scaled])[0]
            proba = model.predict_proba([fv_scaled])[0] if hasattr(model, 'predict_proba') else None
            disease = self.disease_encoder.inverse_transform([pred_enc])[0]
            conf = float(np.max(proba)) if proba is not None else None
            predictions[name] = {'disease': disease, 'confidence': conf}
        return predictions

    # ---------------------------
    # Visualization
    # ---------------------------
    def visualize_results(self):
        # Collect metrics
        names, accs, f1s, train_accs, cv_means = [], [], [], [], []
        for name, res in self.models.items():
            if name == 'GNN':
                continue
            names.append(name)
            accs.append(res['accuracy'])
            f1s.append(res['f1'])
            train_accs.append(res['train_acc'])
            cv_means.append(res['cv_mean'])

        # Basic bar plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].bar(names, train_accs)
        axes[0, 0].set_title('Train Accuracy (balanced train)')
        axes[0, 0].set_ylim(0, 1)

        axes[0, 1].bar(names, accs)
        axes[0, 1].set_title('Test Accuracy')
        axes[0, 1].set_ylim(0, 1)

        axes[1, 0].bar(names, f1s)
        axes[1, 0].set_title('Weighted F1 (Test)')
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].bar(names, cv_means)
        axes[1, 1].set_title('CV Accuracy (Train, SMOTE-applied)')
        axes[1, 1].set_ylim(0, 1)

        for ax in axes.ravel():
            for label in ax.get_xticklabels():
                label.set_rotation(15)
        plt.tight_layout()
        plt.show()

        # Confusion matrix for best test accuracy (non-GNN)
        if names:
            best_name = names[int(np.argmax(accs))]
            cm = confusion_matrix(self.y_test, self.models[best_name]['predictions'])
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix — {best_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            # Classification report
            print(f"\nClassification Report — {best_name}")
            print(classification_report(self.y_test, self.models[best_name]['predictions'],
                                        target_names=np.unique(self.y_test_names)))


# =========================================================
# Utility: Interactive CLI (optional)
# =========================================================
def interactive_disease_prediction(predictor):
    print("=== Disease Prediction System ===")
    s = {}
    s['fever'] = int(input("Fever (1/0): "))
    s['cough'] = int(input("Cough (1/0): "))
    s['fatigue'] = int(input("Fatigue (1/0): "))
    s['difficulty_breathing'] = int(input("Difficulty Breathing (1/0): "))
    s['age'] = int(input("Age: "))
    s['gender'] = int(input("Gender (1=Male,0=Female): "))
    s['blood_pressure'] = int(input("BP (0=Low,1=Normal,2=High): "))
    s['cholesterol_level'] = int(input("Cholesterol (0=Low,1=Normal,2=High): "))
    preds = predictor.predict_disease(s)
    print("\n=== Predictions ===")
    for k, v in preds.items():
        print(f"{k}: {v['disease']} (Conf: {v['confidence']:.2%} if available)")


# =========================================================
# Main: Choose CSV or a Better Synthetic Demo
# =========================================================
def make_demo_dataframe(n=800, seed=42):
    """
    More realistic synthetic data (not repetitive rows).
    Correlates some symptoms with diseases to make learning non-trivial.
    """
    rng = np.random.default_rng(seed)
    diseases = ['Influenza', 'Common Cold', 'Asthma', 'Pneumonia', 'Bronchitis']
    base = pd.DataFrame({
        'Disease': rng.choice(diseases, n, p=[0.22, 0.28, 0.18, 0.17, 0.15]),
        'Age': rng.integers(10, 80, n),
        'Gender': rng.choice(['Male', 'Female'], n),
        'Blood Pressure': rng.choice(['Low', 'Normal', 'High'], n, p=[0.2, 0.6, 0.2]),
        'Cholesterol Level': rng.choice(['Low', 'Normal', 'High'], n, p=[0.2, 0.6, 0.2]),
        'Outcome Variable': rng.choice(['Positive', 'Negative'], n, p=[0.55, 0.45])
    })

    # Symptom probabilities conditioned on disease (to avoid pure randomness)
    probs = {
        'Influenza':        {'Fever': 0.8, 'Cough': 0.7, 'Fatigue': 0.75, 'Difficulty Breathing': 0.25},
        'Common Cold':      {'Fever': 0.25,'Cough': 0.75,'Fatigue': 0.5,  'Difficulty Breathing': 0.1},
        'Asthma':           {'Fever': 0.1, 'Cough': 0.7, 'Fatigue': 0.4,  'Difficulty Breathing': 0.7},
        'Pneumonia':        {'Fever': 0.7, 'Cough': 0.8, 'Fatigue': 0.7,  'Difficulty Breathing': 0.6},
        'Bronchitis':       {'Fever': 0.3, 'Cough': 0.85,'Fatigue': 0.55, 'Difficulty Breathing': 0.4},
    }

    def yes_no(p): return 'Yes' if rng.random() < p else 'No'
    base['Fever'] = [yes_no(probs[d]['Fever']) for d in base['Disease']]
    base['Cough'] = [yes_no(probs[d]['Cough']) for d in base['Disease']]
    base['Fatigue'] = [yes_no(probs[d]['Fatigue']) for d in base['Disease']]
    base['Difficulty Breathing'] = [yes_no(probs[d]['Difficulty Breathing']) for d in base['Disease']]
    return base


def main(use_csv=False, csv_path=None):
    predictor = DiseasePredictor()

    if use_csv:
        # Provide your CSV path (e.g., '/content/your_file.csv')
        predictor.load_and_preprocess_data("backend\Disease_symptom_and_patient_profile_dataset.csv")
    else:
        # Use improved synthetic data (non-repetitive, partially informative)
        df = make_demo_dataframe(n=1000, seed=7)
        predictor.load_and_preprocess_data(df=df)

    X_train, X_test, y_train, y_test = predictor.prepare_splits()
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Unique diseases: {len(np.unique(y_train))}")

    print("\nTraining classical models ...")
    predictor.train_traditional_models()

    print("\nTraining GNN (light) ...")
    try:
        predictor.create_gnn_model(n_neighbors=8, epochs=60, hidden_dim=64, dropout=0.4)
    except Exception as e:
        print(f"GNN training skipped due to error: {e}")

    predictor.visualize_results()

    print("\n=== Model Performance Summary ===")
    for name, info in predictor.models.items():
        if name == 'GNN':
            print(f"{name}: Test Acc = {info['accuracy']:.3f}")
        else:
            print(f"{name}: Train Acc = {info['train_acc']:.3f} | Test Acc = {info['accuracy']:.3f} "
                  f"| F1 = {info['f1']:.3f} | CV = {info['cv_mean']:.3f}")

    return predictor


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    # Toggle use_csv=True and set csv_path to run on your dataset
    predictor = main(use_csv=False, csv_path=None)

    # Example single prediction
    example = {
        'fever': 1,
        'cough': 1,
        'fatigue': 1,
        'difficulty_breathing': 0,
        'age': 35,
        'gender': 1,             # 1=Male, 0=Female
        'blood_pressure': 1,     # 0=Low, 1=Normal, 2=High
        'cholesterol_level': 1   # 0=Low, 1=Normal, 2=High
    }
    print("\n=== Example Prediction (non-GNN models) ===")
    preds = predictor.predict_disease(example)
    for m, r in preds.items():
        conf_str = f"{r['confidence']:.2%}" if r['confidence'] is not None else "NA"
        print(f"{m}: {r['disease']} (Confidence: {conf_str})")

    # For CLI use:
    # interactive_disease_prediction(predictor)
