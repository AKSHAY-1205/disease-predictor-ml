# Disease Predictor Application

A sophisticated disease prediction system using ensemble machine learning models with a modern React frontend.

## Backend Setup (Python Flask)

1. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

2. Install dependencies:
\`\`\`bash
cd backend
pip install -r requirements.txt
\`\`\`

3. Run the Flask app:
\`\`\`bash
python app.py
\`\`\`

The backend will run on `http://localhost:5000`

## Frontend Setup (Next.js React)

1. Install dependencies:
\`\`\`bash
npm install
# or
yarn install
\`\`\`

2. Run the development server:
\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

The frontend will run on `http://localhost:3000`

## Model Training

To train the enhanced ensemble model:
\`\`\`bash
cd backend
python enhanced_disease_predictor.py
\`\`\`

## Features

- **Backend**: Flask API with ensemble ML models (Random Forest, XGBoost, SVM, Gradient Boosting)
- **Frontend**: Modern React UI with glassmorphism design
- **ML Features**: Confidence scoring, model consensus, hyperparameter optimization
- **Data**: Real medical dataset with comprehensive feature engineering

## API Endpoints

- `POST /predict` - Make disease predictions
- `GET /model-info` - Get model information
- `GET /health` - Health check endpoint
