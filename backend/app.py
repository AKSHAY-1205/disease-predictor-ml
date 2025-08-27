# =========================================================
# Flask Backend for Disease Prediction System (Minimal)
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Import your DiseasePredictor
from train_model import DiseasePredictor, make_demo_dataframe

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None

def init_model():
    """Initialize and train the model once."""
    global predictor
    if predictor is None:
        print("Training model...")
        predictor = DiseasePredictor()
        
        # Use synthetic data (change to CSV if needed)
        df = make_demo_dataframe(n=1000, seed=7)
        predictor.load_and_preprocess_data(df=df)
        
        # Train models
        predictor.prepare_splits()
        predictor.train_traditional_models()
        print("Model ready!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Disease Prediction API Running',
        'endpoints': ['/predict', '/health']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded
        init_model()
        
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
        
        # Prepare symptoms
        symptoms = {k: int(data[k]) for k in required}
        
        # Predict
        predictions = predictor.predict_disease(symptoms)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Disease Prediction API...")
    app.run(host='0.0.0.0', port=5000, debug=True)