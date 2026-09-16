FROM python:3.13-slim AS base

WORKDIR /mlops-model-service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS test

COPY requirements-tests.txt .

RUN pip install --no-cache-dir -r requirements-tests.txt

COPY main.py .
COPY routers/ routers/
COPY tests/ tests/

CMD [ "python", "-m", "pytest", "-v" ]

FROM base AS train

COPY requirements-training.txt .

RUN pip install --no-cache-dir -r requirements-training.txt

COPY mlflow_project2.py .
COPY dataset/ dataset/

CMD ["python3", "mlflow_project2.py"]

FROM base AS serve

COPY main.py .
COPY routers/ routers/
COPY model/ model/

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

