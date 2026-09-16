FROM python:3.13-slim AS base

WORKDIR /mlops-model-service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS serve

COPY main.py .
COPY routers/ routers/
COPY model/ model/

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
