# RealEstate

## Local MLflow Tracking (Docker Compose)

This project includes a local-development MLflow stack backed by Postgres and MinIO.

### 1) Configure environment variables

Load environment variables

```bash
source .env
```

If port `9001` is already in use, change MinIO console port before startup:

```bash
export MINIO_CONSOLE_PORT=9011
```

### 2) Start services

```bash
docker compose up -d
```

### 3) Open MLflow UI

```bash
xdg-open "http://localhost:${MLFLOW_UI_PORT:-5000}"
```

### 4) Open MinIO Console

```bash
xdg-open "http://localhost:${MINIO_CONSOLE_PORT:-9001}"
```

Service endpoints:
- MLflow UI: `http://localhost:${MLFLOW_UI_PORT:-5000}`
- MinIO API: `http://localhost:${MINIO_API_PORT:-9000}`
- MinIO Console: `http://localhost:${MINIO_CONSOLE_PORT:-9001}`

Notes:
- MLflow backend store uses Postgres via `MLFLOW_BACKEND_STORE_URI`.
- MLflow artifact store uses MinIO (`S3`) via `MLFLOW_S3_ENDPOINT_URL` and `MLFLOW_S3_BUCKET`.
- The `minio-create-bucket` service initializes the artifacts bucket on startup.
