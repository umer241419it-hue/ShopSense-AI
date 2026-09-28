# ShopSense AI ML Service

FastAPI inference wrapper for the existing scikit-learn model.

Run from repository root:

```bash
pip install -r ml-service/requirements.txt
uvicorn ml-service.main:app --reload --port 8000
```

Endpoints: GET /health, POST /predict, GET /model-info.