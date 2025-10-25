
FROM python:3.11-slim as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    wget \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .


RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt


RUN useradd --create-home appuser
USER appuser
ENV HOME=/home/appuser
WORKDIR /home/appuser/app


COPY --chown=appuser:appuser . .


ENV PORT=8080
EXPOSE 8080


ENV MODEL_PATH=artifacts/model.json
ENV PREPROCESSOR_PATH=artifacts/preprocessor.pkl
ENV PYTHONUNBUFFERED=1


CMD ["gunicorn", "src.api.app:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--config", "gunicorn_conf.py"]
