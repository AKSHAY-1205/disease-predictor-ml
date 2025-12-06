
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
import xgboost as xgb


class EnhancedDiseasePredictor:
    def __init__(self):
        # Trained models & metadata
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        self.disease_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_importance = {}

        # Holdouts for evaluation/plots
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_names = None
        self.y_test_names = None

        # Original preprocessed dataframe
        self.data = None

    def load_data_from_url(self, url):
        """Load data directly from the provided URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.data = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded data from URL. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data from URL: {e}")
            return None

    def load_and_preprocess_data(self, csv_path=None, df=None, url=None):
        """Enhanced data loading with URL support"""
        if url is not None:
            self.data = self.load_data_from_url(url)
        elif df is not None:
            self.data = df.copy()
        else:
            if csv_path is None:
                raise ValueError("Provide either csv_path, df, or url.")
            self.data = pd.read_csv(csv_path)

        if self.data is None:
            raise ValueError("Failed to load data")

        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")

        # Quick look at class distribution
        if 'Disease' in self.data.columns:
            print("\nDisease distribution:")
            print(self.data['Disease'].value_counts())

        self._encode_categorical_features()
        self._handle_missing_values()
        self._advanced_feature_engineering()
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
            if col in self.data.columns and not self.data[col].empty:
                mode_val = self.data[col].mode()
                if len(mode_val) > 0:
                    self.data[col] = self.data[col].fillna(mode_val[0])

    def _advanced_feature_engineering(self):
        """Enhanced feature engineering with more sophisticated derived features"""
        # Basic symptom count
        symptom_cols = [c for c in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'] if c in self.data.columns]
        if symptom_cols:
            self.data['Total_Symptoms'] = self.data[symptom_cols].sum(axis=1)
            
            respiratory_symptoms = [c for c in ['Cough', 'Difficulty Breathing'] if c in self.data.columns]
            if respiratory_symptoms:
                self.data['Respiratory_Score'] = self.data[respiratory_symptoms].sum(axis=1)
        else:
            self.data['Total_Symptoms'] = 0
            self.data['Respiratory_Score'] = 0

        # Enhanced age grouping
        if 'Age' in self.data.columns:
            self.data['Age_Group'] = pd.cut(
                self.data['Age'],
                bins=[-1, 18, 35, 50, 65, 120],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            
            self.data['Elderly_Risk'] = (self.data['Age'] >= 65).astype(int)
            self.data['Young_Adult'] = ((self.data['Age'] >= 18) & (self.data['Age'] <= 35)).astype(int)
        else:
            self.data['Age_Group'] = 0
            self.data['Elderly_Risk'] = 0
            self.data['Young_Adult'] = 0

        # Enhanced risk scoring
        age = self.data['Age'] if 'Age' in self.data.columns else 0
        bp = self.data['Blood Pressure'] if 'Blood Pressure' in self.data.columns else 0
        chol = self.data['Cholesterol Level'] if 'Cholesterol Level' in self.data.columns else 0
        
        self.data['Cardiovascular_Risk'] = (bp * 0.4) + (chol * 0.4) + ((age / 100.0) * 0.2)
        self.data['Overall_Risk_Score'] = (
            (age / 100.0) * 0.3 + 
            (bp * 0.25) + 
            (chol * 0.25) + 
            (self.data['Total_Symptoms'] * 0.2)
        )
        
        if 'Gender' in self.data.columns:
            self.data['Gender_Age_Interaction'] = self.data['Gender'] * (age / 100.0)

    def prepare_splits(self):
        """Enhanced feature preparation with better feature selection"""
        feature_cols = [...]
        self.feature_names = feature_cols

        X = self.data[feature_cols].copy()
        y_disease = self.data['Disease'].copy()

        # ⚠️ Remove diseases with fewer than 2 samples
        disease_counts = y_disease.value_counts()
        rare_diseases = disease_counts[disease_counts < 2].index
        if len(rare_diseases) > 0:
            print(f"Removing {len(rare_diseases)} rare diseases: {list(rare_diseases)}")
            mask = ~y_disease.isin(rare_diseases)
            X = X[mask]
            y_disease = y_disease[mask]

        y_disease_encoded = self.disease_encoder.fit_transform(y_disease)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_disease_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded
        )

        # Fit scaler ONLY on train
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(self.scaler.transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),  columns=feature_cols, index=X_test.index)

        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.y_train_names = self.disease_encoder.inverse_transform(y_train)
        self.y_test_names  = self.disease_encoder.inverse_transform(y_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def handle_class_imbalance(self, X, y):
        """Enhanced SMOTE with better parameters"""
        smote = SMOTE(random_state=42, k_neighbors=3)
        Xb, yb = smote.fit_resample(X, y)
        return Xb, yb

    def optimize_hyperparameters(self, X_train_bal, y_train_bal):
        """Hyperparameter optimization for key models"""
        print("Optimizing hyperparameters...")
        
        optimized_models = {}
        
        # XGBoost optimization
        xgb_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.05, 0.08, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        xgb_grid.fit(X_train_bal, y_train_bal)
        optimized_models['XGBoost'] = xgb_grid.best_estimator_
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 12, 15],
            'min_samples_split': [5, 8, 10],
            'min_samples_leaf': [2, 4, 6]
        }
        
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train_bal, y_train_bal)
        optimized_models['Random Forest'] = rf_grid.best_estimator_
        
        return optimized_models

    def train_enhanced_models(self):
        """Train models with optimization and ensemble methods"""
        assert self.X_train is not None, "Call prepare_splits() first."

        # Apply SMOTE ONLY on training data
        X_train_bal, y_train_bal = self.handle_class_imbalance(self.X_train, self.y_train)

        # Get optimized models
        optimized_models = self.optimize_hyperparameters(X_train_bal, y_train_bal)

        models = {
            **optimized_models,
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale', 
                probability=True, random_state=42
            ),
            'Bagging': BaggingClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        }

        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            print(f"\nTraining {name} ...")
            model.fit(X_train_bal, y_train_bal)

            # Predictions and metrics
            y_train_pred = model.predict(X_train_bal)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

            train_acc = accuracy_score(y_train_bal, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='accuracy')

            print(f"{name} -> Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
                  f"F1: {f1:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

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

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(self.feature_names, model.feature_importances_))

        self.models.update(results)
        
        self._create_ensemble_model(X_train_bal, y_train_bal)
        
        return results

    def _create_ensemble_model(self, X_train_bal, y_train_bal):
        """Create a sophisticated ensemble model"""
        print("\nCreating ensemble model...")
        
        # Select top performing models for ensemble
        model_scores = [(name, info['cv_mean']) for name, info in self.models.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:4]  # Top 4 models
        
        ensemble_estimators = []
        for name, score in top_models:
            ensemble_estimators.append((name.lower().replace(' ', '_'), self.models[name]['model']))
        
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use probability-based voting
        )
        
        self.ensemble_model.fit(X_train_bal, y_train_bal)
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble_model.predict(self.X_test)
        ensemble_acc = accuracy_score(self.y_test, ensemble_pred)
        ensemble_f1 = precision_recall_fscore_support(self.y_test, ensemble_pred, average='weighted')[2]
        
        print(f"Ensemble Model -> Test Acc: {ensemble_acc:.3f} | F1: {ensemble_f1:.3f}")
        
        self.models['Ensemble'] = {
            'model': self.ensemble_model,
            'accuracy': ensemble_acc,
            'f1': ensemble_f1,
            'predictions': ensemble_pred,
            'probabilities': self.ensemble_model.predict_proba(self.X_test)
        }

    def predict_disease_enhanced(self, symptoms_dict):
        """Enhanced prediction with ensemble and confidence scoring"""
        assert self.feature_names, "Train models first."

        # Prepare feature vector
        fv = {name: 0 for name in self.feature_names}
        
        key_map = {
            'fever': 'Fever', 'cough': 'Cough', 'fatigue': 'Fatigue',
            'difficulty_breathing': 'Difficulty Breathing', 'age': 'Age',
            'gender': 'Gender', 'blood_pressure': 'Blood Pressure',
            'cholesterol_level': 'Cholesterol Level'
        }

        for k, v in symptoms_dict.items():
            if k.lower() in key_map:
                fv[key_map[k.lower()]] = v

        # Enhanced derived features
        fv['Total_Symptoms'] = sum([fv.get(c, 0) for c in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']])
        fv['Respiratory_Score'] = sum([fv.get(c, 0) for c in ['Cough', 'Difficulty Breathing']])
        
        age = fv.get('Age', 0)
        fv['Age_Group'] = 0 if age <= 18 else (1 if age <= 35 else (2 if age <= 50 else (3 if age <= 65 else 4)))
        fv['Elderly_Risk'] = 1 if age >= 65 else 0
        fv['Young_Adult'] = 1 if 18 <= age <= 35 else 0
        
        bp = fv.get('Blood Pressure', 0)
        chol = fv.get('Cholesterol Level', 0)
        fv['Cardiovascular_Risk'] = (bp * 0.4) + (chol * 0.4) + ((age / 100.0) * 0.2)
        fv['Overall_Risk_Score'] = (age / 100.0) * 0.3 + (bp * 0.25) + (chol * 0.25) + (fv['Total_Symptoms'] * 0.2)
        fv['Gender_Age_Interaction'] = fv.get('Gender', 0) * (age / 100.0)

        # Scale features
        fv_vec = [fv[name] for name in self.feature_names]
        fv_scaled = self.scaler.transform([fv_vec])[0]

        predictions = {}
        confidence_scores = []
        
        for name, m in self.models.items():
            model = m['model']
            pred_enc = model.predict([fv_scaled])[0]
            proba = model.predict_proba([fv_scaled])[0] if hasattr(model, 'predict_proba') else None
            disease = self.disease_encoder.inverse_transform([pred_enc])[0]
            conf = float(np.max(proba)) if proba is not None else None
            
            predictions[name] = {
                'disease': disease, 
                'confidence': conf,
                'probabilities': proba.tolist() if proba is not None else None
            }
            
            if conf is not None:
                confidence_scores.append(conf)

        disease_votes = {}
        for pred in predictions.values():
            disease = pred['disease']
            disease_votes[disease] = disease_votes.get(disease, 0) + 1
        
        consensus_disease = max(disease_votes.items(), key=lambda x: x[1])[0]
        consensus_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        predictions['Consensus'] = {
            'disease': consensus_disease,
            'confidence': consensus_confidence,
            'vote_distribution': disease_votes
        }
        
        return predictions

    def visualize_enhanced_results(self):
        """Enhanced visualization with feature importance and model comparison"""
        # Model performance comparison
        names, accs, f1s, cv_means = [], [], [], []
        for name, res in self.models.items():
            names.append(name)
            accs.append(res['accuracy'])
            f1s.append(res.get('f1', 0))
            cv_means.append(res.get('cv_mean', res['accuracy']))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Performance metrics
        axes[0, 0].bar(names, accs, color='skyblue')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(names, f1s, color='lightgreen')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Feature importance (if available)
        if self.feature_importance:
            # Average feature importance across models
            avg_importance = {}
            for feature in self.feature_names:
                importances = [imp.get(feature, 0) for imp in self.feature_importance.values()]
                avg_importance[feature] = np.mean(importances)
            
            features = list(avg_importance.keys())
            importance_values = list(avg_importance.values())
            
            axes[1, 0].barh(features, importance_values, color='coral')
            axes[1, 0].set_title('Average Feature Importance')
            axes[1, 0].set_xlabel('Importance')

        # Model ranking
        model_ranking = sorted(zip(names, accs), key=lambda x: x[1], reverse=True)
        rank_names, rank_scores = zip(*model_ranking)
        
        axes[1, 1].bar(rank_names, rank_scores, color='gold')
        axes[1, 1].set_title('Model Ranking by Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Best model confusion matrix
        best_model_name = max(self.models.items(), key=lambda x: x[1]['accuracy'])[0]
        if 'predictions' in self.models[best_model_name]:
            cm = confusion_matrix(self.y_test, self.models[best_model_name]['predictions'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.disease_encoder.classes_,
                       yticklabels=self.disease_encoder.classes_)
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

def main():
    """Enhanced main function with real dataset"""
    predictor = EnhancedDiseasePredictor()
    
    dataset_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Disease_symptom_and_patient_profile_dataset-clkE6NGbzNVnVTeFwrOfVzQCMstJ9W.csv"
    
    print("Loading dataset from URL...")
    predictor.load_and_preprocess_data(url=dataset_url)
    
    print("\nPreparing train/test splits...")
    X_train, X_test, y_train, y_test = predictor.prepare_splits()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Unique diseases: {len(np.unique(y_train))}")

    print("\nTraining enhanced models with optimization...")
    predictor.train_enhanced_models()

    print("\nVisualizing results...")
    predictor.visualize_enhanced_results()

    print("\n=== Enhanced Model Performance Summary ===")
    for name, info in predictor.models.items():
        acc = info['accuracy']
        f1 = info.get('f1', 0)
        cv = info.get('cv_mean', acc)
        print(f"{name}: Test Acc = {acc:.3f} | F1 = {f1:.3f} | CV = {cv:.3f}")

    # Example prediction with enhanced features
    example = {
        'fever': 1, 'cough': 1, 'fatigue': 1, 'difficulty_breathing': 0,
        'age': 45, 'gender': 1, 'blood_pressure': 2, 'cholesterol_level': 1
    }
    
    print("\n=== Enhanced Prediction Example ===")
    preds = predictor.predict_disease_enhanced(example)
    for model_name, result in preds.items():
        if model_name == 'Consensus':
            print(f"\n{model_name}: {result['disease']} (Avg Confidence: {result['confidence']:.2%})")
            print(f"Vote Distribution: {result['vote_distribution']}")
        else:
            conf_str = f"{result['confidence']:.2%}" if result['confidence'] is not None else "N/A"
            print(f"{model_name}: {result['disease']} (Confidence: {conf_str})")

    return predictor

if __name__ == "__main__":
    predictor = main()