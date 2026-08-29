# mlops-model-service

XGBoost model trained on the Titanic dataset, served via FastAPI and containerized with Docker.

## Stack

- **scikit-learn + XGBoost** — model training with hyperparameter tuning via Optuna
- **MLflow** — experiment tracking and model registry
- **FastAPI** — REST API for model serving
- **Docker** — containerization

## Project Structure

```
mlops-model-service/
├── main.py               # FastAPI app entry point
├── routers/
│   └── predict.py        # /predict endpoint
├── mlflow_project2.py    # Training script
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Run Locally with Docker

```bash
# Build the image
docker build -t mlops-model-service .

# Run the container
docker run -p 8000:8000 mlops-model-service
```

## API

### `POST /predict`

Send a JSON body with the passenger features and get a survival prediction back.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "sex": 1}'
```

## Training

To retrain the model and log experiments to MLflow:

```bash
python mlflow_project2.py
```

View results in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflowP2.db
```
