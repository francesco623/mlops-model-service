# mlops-model-service

![CI](https://github.com/francesco623/mlops-model-service/actions/workflows/ci.yml/badge.svg)

XGBoost model trained on the Titanic dataset, served via FastAPI and containerized with Docker. Tests and a smoke test against the running API run automatically on every push.

Built as the Level 2 checkpoint of my MLOps learning roadmap.

## Stack

- **scikit-learn + XGBoost**: model training, with hyperparameter tuning via Optuna
- **MLflow**: experiment tracking and model packaging
- **FastAPI**: REST API for model serving
- **Docker + Docker Compose**: one multi-stage Dockerfile for serving, testing and training
- **pytest**: API tests with a fake model (no real model needed)
- **GitHub Actions**: CI that runs the tests and a smoke test on every push

## Project Structure

```
mlops-model-service/
├── main.py                    # FastAPI app entry point
├── routers/
│   └── predict.py             # /predict endpoint, model loaded lazily via dependency
├── model/                     # Trained model (MLflow format), shipped with the repo
├── dataset/
│   ├── data.csv               # Titanic features (raw)
│   └── target.csv             # Titanic target (survived)
├── mlflow_project2.py         # Training script: Optuna search, then saves best model to model/
├── scripts/
│   └── download_data.py       # Optional: re-download the dataset from OpenML
├── tests/
│   └── test_predict.py        # API tests
├── .github/workflows/ci.yml   # CI: tests + smoke test
├── Dockerfile                 # Stages: base, test, train, serve
├── docker-compose.yaml        # Services: api, test, train
├── requirements.txt           # Serving dependencies
├── requirements-tests.txt     # + pytest, httpx
└── requirements-training.txt  # + scikit-learn, optuna
```

## Quickstart

The only requirement is Docker. The trained model is already in the repo, so no training is needed to run the API.

```bash
docker compose up --build
```

Then, from another terminal:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "sex": 0}'
```

A 1st-class female passenger, so the expected answer is `"survived": 1`:

```json
{"survived": 1, "probability": <value between 0.5 and 1>}
```

## API

### `POST /predict`

| Field    | Type  | Required | Notes              |
|----------|-------|----------|--------------------|
| `pclass` | float | yes      | 1, 2 or 3          |
| `age`    | float | no       | can be missing     |
| `sibsp`  | float | yes      | siblings/spouses aboard |
| `parch`  | float | yes      | parents/children aboard |
| `fare`   | float | yes      |                    |
| `body`   | float | no       | almost always missing |
| `sex`    | int   | yes      | 1 = male, 0 = female |

Returns `survived` (0 or 1, threshold 0.5) and `probability`. A missing required field returns `422`.

## Run the Tests

```bash
docker compose run --rm --build test
```

The tests replace the real model with a fake one (via FastAPI's `dependency_overrides`), so they don't need `model/`.

CI runs the same command on every push, then starts the real API container and sends a request to `/predict` as a smoke test.

## Retrain the Model

Training runs in a container too, with fixed seeds, so the result is reproducible.

```bash
# 1. Retrain: writes the new model into model/
docker compose run --rm --build train

# 2. Rebuild the API so the image includes the new model
docker compose up --build

# 3. Commit the new model
git add model/ && git commit -m "Retrain model"
```

## Dataset

Titanic dataset from OpenML (`name="titanic"`, `version=1`), stored raw in `dataset/` so training works offline. To download it again:

```bash
pip install scikit-learn pandas
python scripts/download_data.py
```
