# ShopSense AI

**AI-Powered Online Shopper Purchase Intention Prediction & Analytics Platform**

ShopSense AI combines an academic machine-learning experiment with a complete web application:

- **React + Vite** frontend
- **Node.js + Express** REST API
- **MongoDB + Mongoose** application persistence
- **FastAPI** ML inference service
- **scikit-learn** training and inference pipeline
- UCI Online Shoppers dataset, preprocessing, EDA, evaluation, notebook and reports

## Dual-Model Architecture

The project maintains two model contexts:
1. **Academic Benchmark Model (`models/best_model.joblib`)**: Uses the complete historical UCI 17-feature dataset (including retrospective Google Analytics attribution features like `PageValues`, `BounceRates`, `ExitRates` and anonymized technical codes) for reproducible dataset benchmarking, viva defense, and research reporting.
2. **Production Inference Model (`models/production_model.joblib`)**: Trained exclusively on features that are **genuinely observable in real-time** during an active browsing session.

```
React / Vite :5173
      |
      v
Express API :5000  -----> MongoDB :27017
      |
      v
FastAPI ML Service :8000
      |
      +---> models/production_model.joblib (Active Real-Time Inference: 10 Observable Features)
      |
      +---> models/best_model.joblib       (Academic Research Benchmark: 17 UCI Features)
```

### Technical Note on Dual-Model Architecture
> "The project maintains two model contexts. The academic benchmark model uses the complete historical UCI feature set for reproducible dataset benchmarking. The production inference model uses only features that are genuinely observable during an active browsing session. This separation prevents retrospective attribution variables and anonymized dataset identifiers from being fabricated as user inputs."

*Note on Predictive Performance*: The production model is expected to have lower predictive performance ($F_1 \approx 0.38-0.40$ vs. $F_1 \approx 0.68$) because it deliberately removes retrospective/derived attribution features (especially `PageValues`, which is calculated post-session in Google Analytics).

### Observable Production Features (10 Total):
- **Numerical (7)**: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `SpecialDay`
- **Categorical (3)**: `Month`, `VisitorType`, `Weekend`
- **Excluded**: `PageValues`, `BounceRates`, `ExitRates`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`

## Main flow

1. User registers or signs in.
2. React sends an authenticated request containing strictly real-time observable session fields to Express.
3. Express validates the 10 production features and forwards the payload to FastAPI.
4. FastAPI validates the schema and runs the persisted `production_model.joblib` pipeline.
5. Express stores the input and prediction in MongoDB.
6. React displays the result and prediction history.
7. The Model page displays transparency reports for both the production model and academic benchmark.

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

## ML experiment & Model Training

### Academic Benchmark Model (Historical 17 UCI Features)
Install the research dependencies and run:
```bash
pip install -r requirements.txt
python run_project.py
```
This runs data acquisition, validation, EDA, feature engineering, model comparison, class-imbalance analysis, hyperparameter tuning, evaluation and inference checks for the academic benchmark artifact (`models/best_model.joblib`).

### Production Inference Model (10 Real-Time Observable Features)
To retrain and evaluate the production model pipeline:
```bash
python src/train_production.py
```
This trains candidate classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) using only observable browsing metrics, performs hyperparameter tuning, and saves the calibrated inference pipeline to `models/production_model.joblib` and metrics to `reports/production_model_comparison.csv`.

## Offline-Capable Architecture
ShopSense AI is entirely self-contained and operates in fully air-gapped / offline environments:
- **Zero Cloud AI Dependencies**: All inference is executed locally using serialized scikit-learn pipelines; no third-party LLM or cloud inference APIs are queried.
- **Local Asset Bundling**: Fonts and CSS icons are bundled locally with Vite; no Google Fonts or external CDNs are requested at runtime.
- **Local Persistence & Service Mesh**: MongoDB, Node.js Express, and FastAPI run entirely on local network loops (`127.0.0.1`).

## Testing & Quality Assurance

Run the complete automated test suite (pipeline validation and API integration tests):
```bash
python -m pytest -q
```
The test suite validates:
- UCI dataset schema integrity and preprocessing
- Academic benchmark model loading and 17-feature inference
- Production model loading and strict 10-feature inference
- Backend and ML service health
- Authentication, JWT verification, and unauthorized request rejection
- Input validation (negative numbers, invalid months, non-boolean values)
- End-to-end prediction persistence and analytics aggregation

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
models/         persisted trained pipelines (production_model.joblib & best_model.joblib)
notebooks/      research notebook
reports/        metrics, figures, evaluation reports, and viva questions
tests/          dual-model ML pipeline and API integration tests
```
