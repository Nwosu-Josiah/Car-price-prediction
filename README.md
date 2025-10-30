# 🚗 Car Price Prediction API (Production-Ready ML System)

A **production-grade machine learning project** for predicting used car prices.  
This project demonstrates **end-to-end ML deployment** — including **data preprocessing, model training, optimization, containerization, versioning, load testing, monitoring, and serving predictions via a FastAPI microservice**.

---

## 🧠 Key Capabilities

✅ **Containerized ML model with Docker**  
✅ **Prediction serving via FastAPI API**  
✅ **Load testing with k6 / Locust**  
✅ **Monitoring with Prometheus + Google Cloud Monitoring (Stackdriver)**  
✅ **Data & model versioning with DVC**  
✅ **Hyperparameter tuning with Optuna**  
✅ **Optimized inference latency & auto-scaling via GCP Cloud Run**

---

## 🏗️ Project Structure

```
Car-price-prediction/
│
├── artifacts/
│   ├── model.json                  
│   └── preprocessor.pkl
│
├── datasets/
│   └── raw_data_sampled.csv        
│
├── notebooks/
│   ├──Cars_prediction.ipynb
│   └──Exploratory_analysis.ipynb
│
├── reports/
│   └──metrics.json
│
├── src/
│   ├── api/
│   │   └── app.py 
│   ├── data_preprocessing/
│   │   └── preprocessor.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── model_factory.py
│   │   └── xgboost_model.py
│   ├── monitoring/
│       ├── alert_rules.yaml
│       ├── dashboard.json
│       ├── logger.py
│       ├── metrics.py       
│       └── prometheus.yaml
│   ├── prediction/
│   │   └── predictor.py          
│   └── training/
│      ├── prepare_data.py              
│      └── trainer.py            
│
├── tests/
│   ├── load_test.js              
│   ├── load_test.yaml
│   ├── test_api.py
│   ├── test_logger.py
│   ├── test_model_factory.py
│   ├── test_preprocessor.py               
│   └── test_trainer.py               
│
├── Dockerfile                      
├── cloudbuild.yaml                 
├── dvc.yaml                        
├── dvc.lock  
├── gunicorn_conf.py
├── app.py #For Streamlit dashboard                       
├── requirements.txt                
└── README.md                       
```

---

## ⚙️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.11 |
| **ML Framework** | XGBoost (low-level API) |
| **Preprocessing** | Scikit-learn |
| **Hyperparameter Tuning** | Optuna |
| **Serving** | FastAPI + Uvicorn |
| **Containerization** | Docker |
| **CI/CD** | Google Cloud Build |
| **Deployment** | Cloud Run |
| **Versioning** | DVC + Git |
| **Monitoring** | Prometheus |
| **Load Testing** | k6 |

---

## 🚀 Setup & Local Development

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nwosu-Josiah/Car-price-prediction.git
cd Car-price-prediction
```

### 2️⃣ Create a Virtual Environment

```bash
conda create -n carprice python=3.11 -y
conda activate carprice
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python src/training/trainer.py
```

---

## 🧩 Running the API Locally

### 1️⃣ Build the Docker Image

```bash
docker build -t carprice-api .
```

### 2️⃣ Run the Container

```bash
docker run -p 8080:8080 carprice-api
```

### 3️⃣ Access the API

- **Docs:** [http://localhost:8080/docs](http://localhost:8080/docs)  
- **Health Check:** [http://localhost:8080/health](http://localhost:8080/health)

---

## 🧪 Load Testing (Inference Benchmarking)

### Run k6 Load Test

```bash
k6 run tests/load_test.js
```

### Example Metrics Collected
- Average Inference Latency (ms)
- Throughput (req/sec)
- Error Rate (%)

---

## 📈 Monitoring Setup

**Prometheus** monitors:
- Request volume
- API response time
- Model inference latency
- Error rates

### Prometheus Configuration (`prometheus.yaml`)

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'carprice-api'
    static_configs:
      - targets: ['localhost:8080']
```

### Example Alert Rules (`alert_rules.yaml`)

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighLatency
        expr: http_request_duration_seconds_avg > 1
        for: 1m
        labels:
          severity: warning
        annotations:
          description: "API latency is above 1s for over 1 minute."
```

---

## 📦 DVC Versioning Setup

### Initialize DVC
```bash
dvc init
dvc remote add -d gcs_remote gs://carprice-artifacts
```

### Track Data and Models
```bash
dvc add datasets/raw_data_sampled.csv
dvc add artifacts/model.json
git add datasets/raw_data_sampled.csv.dvc artifacts/model.json.dvc
git commit -m "Track dataset and model with DVC"
```

### Reproduce Full Pipeline
```bash
dvc repro
```

---

## ☁️ Deploy on Google Cloud Run

1️⃣ **Authenticate:**
```bash
gcloud auth login
gcloud auth configure-docker us-east1-docker.pkg.dev
```

2️⃣ **Build & Push Image:**
```bash
docker build -t us-east1-docker.pkg.dev/YOUR_PROJECT/carprice-repo/carprice-api:latest .
docker push us-east1-docker.pkg.dev/YOUR_PROJECT/carprice-repo/carprice-api:latest
```

3️⃣ **Deploy:**
```bash
gcloud run deploy carprice-api   --image us-east1-docker.pkg.dev/YOUR_PROJECT/carprice-repo/carprice-api:latest   --region us-east1   --platform managed   --allow-unauthenticated
```

4️⃣ **Test in Postman:**
- Method: `POST`
- URL: `https://<cloud-run-url>/predict`
- Body (JSON):
```json
{
  "year": 2019,
  "odometer": 42000,
  "condition": "Good",
  "fuel": "Gas",
  "transmission": "Automatic",
  "manufacturer": "Toyota",
  "model": "Highlander"
}
```

---

## 📊 Example Results

| Metric | Description | Example |
|--------|--------------|----------|
| **RMSE** | Root Mean Squared Error (on log scale) | 2.45 |
| **Inference Latency (95%)** | Time for prediction | < 300ms |
| **API Uptime** | Availability | 99.9% |
| **Throughput** | Requests per second | 50+ RPS |

---

## 👨‍💻 Author

**Josiah Nwosu**  
Machine Learning Engineer   
📧 Email: mwosujosiah20@gmail.com  
🔗 GitHub: [Nwosu-Josiah](https://github.com/Nwosu-Josiah)

---

## 📜 License
This project is licensed under the **MIT License**.
