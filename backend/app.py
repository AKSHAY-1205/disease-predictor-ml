# # =========================================================
# # Flask Backend for Disease Prediction System (Minimal)
# # =========================================================

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import warnings
# warnings.filterwarnings('ignore')

# # Import your DiseasePredictor
# from train_model import DiseasePredictor, make_demo_dataframe

# app = Flask(__name__)
# CORS(app)

# # Global predictor instance
# predictor = None

# def init_model():
#     """Initialize and train the model once."""
#     global predictor
#     if predictor is None:
#         print("Training model...")
#         predictor = DiseasePredictor()
        
#         # Use synthetic data (change to CSV if needed)
#         df = make_demo_dataframe(n=1000, seed=7)
#         predictor.load_and_preprocess_data(df=df)
        
#         # Train models
#         predictor.prepare_splits()
#         predictor.train_traditional_models()
#         print("Model ready!")

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         'status': 'Disease Prediction API Running',
#         'endpoints': ['/predict', '/health']
#     })

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'healthy'})

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Ensure model is loaded
#         init_model()
        
#         # Get input data
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
        
#         # Required fields
#         required = ['fever', 'cough', 'fatigue', 'difficulty_breathing', 
#                    'age', 'gender', 'blood_pressure', 'cholesterol_level']
        
#         # Check missing fields
#         missing = [f for f in required if f not in data]
#         if missing:
#             return jsonify({'error': f'Missing fields: {missing}'}), 400
        
#         # Prepare symptoms
#         symptoms = {k: int(data[k]) for k in required}
        
#         # Predict
#         predictions = predictor.predict_disease(symptoms)
        
#         return jsonify({
#             'status': 'success',
#             'predictions': predictions
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     print("Starting Disease Prediction API...")
#     app.run(host='0.0.0.0', port=5000, debug=True)
# =========================================================
# Flask Backend for Enhanced Disease Prediction System
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
ensemble_model = None
label_encoder = None
scaler = None
feature_columns = None

class EnhancedDiseasePredictor:
    def __init__(self):
        self.ensemble_model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_trained = False
        
    def load_and_preprocess_data(self):
        """Load and preprocess the real dataset"""
        try:
            url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Disease_symptom_and_patient_profile_dataset-clkE6NGbzNVnVTeFwrOfVzQCMstJ9W.csv"
            df = pd.read_csv(url)
            
            # Preprocess the data
            # Convert Yes/No to 1/0 for symptoms
            symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
            for col in symptom_cols:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
            
            # Convert categorical variables
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
            df['Blood Pressure'] = df['Blood Pressure'].map({'High': 2, 'Normal': 1, 'Low': 0})
            df['Cholesterol Level'] = df['Cholesterol Level'].map({'High': 2, 'Normal': 1, 'Low': 0})
            
            # Convert Age to numeric
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            
            df['Risk_Score'] = (df['Age'] / 100) + (df['Blood Pressure'] * 0.3) + (df['Cholesterol Level'] * 0.2)
            df['Symptom_Count'] = df[['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']].sum(axis=1)
            df['Age_Risk'] = (df['Age'] > 60).astype(int)
            
            # Prepare features and target
            feature_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
                           'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
                           'Risk_Score', 'Symptom_Count', 'Age_Risk']
            
            X = df[feature_cols]
            y = df['Disease']
            
            # Encode target variable
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            self.feature_columns = feature_cols
            
            return X_scaled, y_encoded, df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def train_model(self):
        """Train the enhanced ensemble model"""
        try:
            X, y, df = self.load_and_preprocess_data()
            if X is None:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            # Create ensemble with optimized parameters
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6,
                random_state=42
            )
            
            svm_model = SVC(
                C=10, gamma='scale', kernel='rbf', probability=True,
                random_state=42
            )
            
            lr_model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )
            
            # Create voting ensemble
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model),
                    ('svm', svm_model),
                    ('lr', lr_model)
                ],
                voting='soft'
            )
            
            # Train the ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.ensemble_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Enhanced Model Accuracy: {accuracy:.4f}")
            self.model_trained = True
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_disease(self, symptoms_dict):
        """Make prediction with confidence scores"""
        try:
            if not self.model_trained:
                return {"error": "Model not trained"}
            
            features = [
                symptoms_dict.get('fever', 0),
                symptoms_dict.get('cough', 0),
                symptoms_dict.get('fatigue', 0),
                symptoms_dict.get('difficulty_breathing', 0),
                symptoms_dict.get('age', 30),
                symptoms_dict.get('gender', 0),  # 0=Female, 1=Male
                symptoms_dict.get('blood_pressure', 1),  # 0=Low, 1=Normal, 2=High
                symptoms_dict.get('cholesterol_level', 1)  # 0=Low, 1=Normal, 2=High
            ]
            
            # Calculate enhanced features
            age = features[4]
            bp = features[6]
            chol = features[7]
            
            risk_score = (age / 100) + (bp * 0.3) + (chol * 0.2)
            symptom_count = sum(features[:4])
            age_risk = 1 if age > 60 else 0
            
            features.extend([risk_score, symptom_count, age_risk])
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and probabilities
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # Get disease name
            disease_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence score
            confidence = float(max(probabilities))
            
            individual_predictions = {}
            for name, model in self.ensemble_model.named_estimators_.items():
                pred = model.predict(features_scaled)[0]
                pred_name = self.label_encoder.inverse_transform([pred])[0]
                individual_predictions[name] = pred_name
            
            return {
                "predicted_disease": disease_name,
                "confidence": confidence,
                "individual_models": individual_predictions,
                "risk_assessment": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

# Global predictor instance
predictor = None

def init_model():
    """Initialize and train the enhanced model once."""
    global predictor
    if predictor is None:
        print("Training enhanced ensemble model...")
        predictor = EnhancedDiseasePredictor()
        
        success = predictor.train_model()
        if success:
            print("Enhanced model ready!")
        else:
            print("Model training failed!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Enhanced Disease Prediction API Running',
        'version': '2.0',
        'features': ['Ensemble Learning', 'Confidence Scores', 'Risk Assessment'],
        'endpoints': ['/predict', '/health', '/model-info']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.model_trained if predictor else False
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    if predictor and predictor.model_trained:
        return jsonify({
            'model_type': 'Ensemble (RF + GB + SVM + LR)',
            'features': predictor.feature_columns,
            'diseases': list(predictor.label_encoder.classes_)
        })
    return jsonify({'error': 'Model not trained'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded
        init_model()
        
        if not predictor or not predictor.model_trained:
            return jsonify({'error': 'Model not available'}), 500
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required = ['fever', 'cough', 'fatigue', 'difficulty_breathing', 
                   'age', 'gender', 'blood_pressure', 'cholesterol_level']
        
        # Check missing fields
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        try:
            symptoms = {}
            symptoms['fever'] = int(data['fever'])
            symptoms['cough'] = int(data['cough'])
            symptoms['fatigue'] = int(data['fatigue'])
            symptoms['difficulty_breathing'] = int(data['difficulty_breathing'])
            symptoms['age'] = float(data['age'])
            symptoms['gender'] = int(data['gender'])
            symptoms['blood_pressure'] = int(data['blood_pressure'])
            symptoms['cholesterol_level'] = int(data['cholesterol_level'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid data types provided'}), 400
        
        # Validate ranges
        if not (0 <= symptoms['age'] <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        
        # Predict
        result = predictor.predict_disease(symptoms)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Enhanced Disease Prediction API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
