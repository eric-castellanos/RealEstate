FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1

RUN pip install mlflow==2.14.3 boto3 psycopg2-binary

EXPOSE 5000

ENTRYPOINT ["mlflow"]
