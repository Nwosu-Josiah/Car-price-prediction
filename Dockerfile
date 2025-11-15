FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pre-install pip tools for faster wheel builds
RUN python -m pip install --upgrade pip setuptools wheel


RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


RUN useradd --create-home appuser
USER appuser

WORKDIR /home/appuser/app

COPY --chown=appuser:appuser . .

# XGBoost threading control (VERY IMPORTANT)
ENV OMP_NUM_THREADS=1
ENV XGBOOST_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENV PORT=8080
EXPOSE 8080

# Model paths
ENV MODEL_PATH=artifacts/model.json
ENV PREPROCESSOR_PATH=artifacts/preprocessor.pkl

ENV PYTHONUNBUFFERED=1

# Use EXACT tuning for ML inference
CMD ["gunicorn", "src.api.app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--threads", "1", \
     "--timeout", "180", \
     "--bind", "0.0.0.0:8080"]
