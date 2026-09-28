# ShopSense AI

**AI-Powered Online Shopper Purchase Intention Prediction & Analytics Platform**

ShopSense AI combines an academic machine-learning experiment with a complete web application:

- **React + Vite** frontend
- **Node.js + Express** REST API
- **MongoDB + Mongoose** application persistence
- **FastAPI** ML inference service
- **scikit-learn** training and inference pipeline
- UCI Online Shoppers dataset, preprocessing, EDA, evaluation, notebook and reports

## Architecture

```
React / Vite :5173
      |
      v
Express API :5000  -----> MongoDB :27017
      |
      v
FastAPI ML Service :8000
      |
      v
models/best_model.joblib
      |
      v
UCI Online Shoppers Purchasing Intention dataset
```

MongoDB stores application users and prediction history. The UCI CSV remains the reproducible ML training source.

## Main flow

1. User registers or signs in.
2. React sends an authenticated request to Express.
3. Express validates the request and forwards the session to FastAPI.
4. FastAPI validates the feature ranges and runs the persisted scikit-learn pipeline.
5. Express stores the input and prediction in MongoDB.
6. React displays the result and prediction history.
7. The Model page reads the stored experimental metrics.

## Local setup

### Requirements

- Node.js 20+
- Python 3.10+
- MongoDB 7+
- Git

### 1. ML service

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r ml-service/requirements.txt
uvicorn main:app --app-dir ml-service --reload --port 8000
```

Health: `http://localhost:8000/health`

### 2. Express backend

```bash
cd backend
npm install
# copy .env.example to .env and set a JWT_SECRET of at least 32 characters
npm run dev
```

Health: `http://localhost:5000/api/health`

### 3. React frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

If the API is not local, set `VITE_API_URL` in `frontend/.env`.

## ML experiment

Install the research dependencies and run:

```bash
pip install -r requirements.txt
python run_project.py
```

This runs data acquisition, validation, EDA, feature engineering, model comparison, class-imbalance analysis, hyperparameter tuning, evaluation and inference checks.

The repository also contains an executed research notebook and academic reports.

## Security and reliability

- Passwords are hashed with bcrypt.
- JWT authentication protects prediction, history and analytics endpoints.
- Password hashes are never returned in normal user responses.
- CORS is restricted to configured client origins.
- Backend rejects invalid auth payloads and uses a centralized error response.
- FastAPI validates prediction ranges before model inference.
- ML service failures are mapped to actionable API responses.
- Prediction history is scoped to the authenticated user.
- Secrets and local environment files are excluded from Git.

## Academic coverage

**Data preprocessing → EDA → feature engineering → model development → model evaluation → result interpretation → REST inference → MongoDB persistence → React visualization**

## Repository structure

```
backend/        Express API and MongoDB models
frontend/       React application
ml-service/     FastAPI inference API
src/            ML training/preprocessing/evaluation code
data/           raw and processed dataset artifacts
models/         persisted trained pipeline
notebooks/      research notebook
reports/        metrics, figures, write-up and viva questions
tests/          ML pipeline tests
```
