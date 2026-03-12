markdown# Titanic ML Pipeline — Airflow + MLflow

A machine learning pipeline for Titanic survival prediction using Apache Airflow for orchestration and MLflow for experiment tracking.

## Project Structure
```
titanic-airflow-mlflow/
├── dags/
│   └── titanic_ml_pipeline.py
├── data/
│   └── titanic.csv
├── logs/
├── mlflow_artifacts/
└── docker-compose.yaml
```

## Prerequisites

- Docker Desktop installed and running
- Titanic dataset (`titanic.csv`) placed in the `data/` folder

## Getting Started

### Start all services
```bash
docker compose up -d
```

### Stop all services
```bash
docker compose down
```

## Accessing the UIs

| Service  | URL                   | Credentials       |
|----------|-----------------------|-------------------|
| Airflow  | http://localhost:8080 | admin / admin     |
| MLflow   | http://localhost:5000 | No login required |

## Running the Pipeline

1. Go to **http://localhost:8080**
2. Find `titanic_ml_pipeline` DAG
3. Click **Trigger DAG w/ config**
4. Pass hyperparameters as JSON:
```json
{"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}
```

### Recommended Runs for Comparison

| Run | n_estimators | max_depth | min_samples_split |
|-----|-------------|-----------|-------------------|
| 1   | 100         | 5         | 2                 |
| 2   | 200         | 10        | 5                 |
| 3   | 50          | 3         | 10                |

## DAG Tasks

| Task               | Description                              |
|--------------------|------------------------------------------|
| `ingest`           | Loads raw CSV data                       |
| `validate`         | Drops rows with missing target           |
| `impute_age`       | Fills missing Age with median            |
| `impute_embarked`  | Fills missing Embarked with 'S'          |
| `features`         | Creates FamilySize feature               |
| `encode`           | One-hot encodes Sex and Embarked         |
| `train`            | Trains RandomForestClassifier            |
| `evaluate`         | Computes accuracy                        |
| `branch`           | Routes to register or reject (>0.75 acc) |
| `register_model`   | Logs metrics and model to MLflow         |
| `reject_model`     | Prints rejection message                 |
| `done`             | Marks pipeline complete                  |

## Metrics Tracked in MLflow

- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC

## Services

| Service            | Image                          | Port |
|--------------------|--------------------------------|------|
| Airflow Webserver  | apache/airflow:2.9.1           | 8080 |
| Airflow Scheduler  | apache/airflow:2.9.1           | —    |
| MLflow             | ghcr.io/mlflow/mlflow:v2.11.3  | 5000 |
| PostgreSQL         | postgres:13                    | 5432 |
