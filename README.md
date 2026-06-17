# High-Throughput Financial Fraud Detection Pipeline

An end-to-end, production-grade machine learning pipeline designed to detect fraudulent transactions under extreme class imbalance. This project moves beyond isolated Jupyter notebooks by embedding an optimized **XGBoost** classifier inside a containerized **FastAPI** application, complete with automated data drift monitoring using **Evidently AI**.

🔴 **Live:** [https://fraud-detection-api-wxa6.onrender.com](https://fraud-detection-api-wxa6.onrender.com)

---

## 🛠️ Key Technical Features

* **Extreme Class Imbalance Handling:** Trained on the IEEE-CIS dataset (~96.5% legitimate, ~3.5% fraud) using stratified cross-validation and tuned `scale_pos_weight` parameters to prevent majority-class bias.
* **Advanced Feature Engineering:** Engineers transaction frequency, time-based components (Hour of Day, Day of Week), behavioral aggregations (deviation from card-level transaction means), and rolling time deltas to capture complex fraud patterns.
* **Rigorous Hyperparameter Optimization:** Uses **Optuna** to search the XGBoost parameter space over 30 trials across 5 stratified folds, targeting **Precision-Recall AUC (PR-AUC)** rather than standard accuracy, which fails on imbalanced data.
* **Production Serving Engine:** Asynchronous **FastAPI** wrapper with rigid structural data validation enforced via **Pydantic** models.
* **Enterprise Containerization:** Fully containerized via a lean **Docker** configuration optimized for cloud deployment on Render.
* **MLOps Observability:** Integrates **Evidently AI** statistical profiling to monitor incoming feature distributions and data drift against the training baseline.

---

## 📂 Project Repository Structure

```text
├── app.py                  # Asynchronous FastAPI production serving engine
├── train.py                # Kaggle-optimized model training & Optuna tuning script
├── monitor_drift.py        # Evidently AI data drift validation suite
├── Dockerfile              # Lean containerization config optimized for cloud deployment
├── requirements.txt        # Environment dependencies
├── fraud_model.joblib      # Serialized XGBoost model artifact (trained on Kaggle)
└── feature_columns.joblib  # Serialized feature schema for inference alignment
```

---

## 📊 Model Performance

Standard accuracy is a deceptive metric for fraud detection — a naive model that always predicts "Not Fraud" achieves ~96.5% accuracy on the IEEE-CIS dataset while catching **zero fraud cases**.

This model explicitly optimizes for **PR-AUC (Average Precision)**:

| Metric | Score |
|---|---|
| Baseline (random guessing) | ~0.035 (equal to fraud class prevalence) |
| Optimized Model PR-AUC | **0.2638** |

The hyperparameter-tuned model performs over **7× better than random chance** on the positive class, heavily suppressing false negatives (missed fraud) while keeping false positives manageable to preserve user experience.

---

## 🔌 API Documentation & Endpoints

**Interactive Swagger UI:** [https://fraud-detection-api-wxa6.onrender.com/docs](https://fraud-detection-api-wxa6.onrender.com/docs)

### `GET /health`

Validates model readiness and service health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Scores an incoming transaction for fraud risk in real time.

**Sample Request:**

```bash
curl -X POST https://fraud-detection-api-wxa6.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.50,
    "card1": 1234.0,
    "card2": 567.0,
    "addr1": 89.0,
    "dist1": 12.5,
    "C1": 1.0,
    "C2": 2.0,
    "C3": 0.0,
    "C4": 0.0,
    "C5": 1.0,
    "C6": 1.0,
    "C7": 0.0,
    "C8": 0.0,
    "C9": 1.0,
    "C10": 0.0,
    "TransactionHour": 14,
    "TransactionAmt_to_mean_card1": 1.12,
    "TransactionAmt_to_std_card1": 0.45,
    "time_delta_card1": 360.0
  }'
```

**Sample Response:**

```json
{
  "fraud_probability": 0.0241,
  "is_fraud": 0,
  "status": "APPROVED"
}
```

> Any transaction exceeding the risk threshold is automatically flagged as `FLAGGED_FOR_REVIEW` for human triage.

---

## ⚙️ Local Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### 2. Run Locally via Docker

```bash
# Build the Docker image
docker build -t fraud-detection-service:v1 .

# Run the container on port 8000
docker run -p 8000:10000 -e PORT=10000 fraud-detection-service:v1
```

### 3. Verify Local Deployment

Navigate to `http://localhost:8000/docs` to interact with the live Swagger documentation.

---

## 📉 Production Observability (Data Drift)

To evaluate whether production input distributions have diverged from the training baseline, run the drift analysis engine:

```bash
python monitor_drift.py
```

This executes Kolmogorov–Smirnov statistical tests across all model features and generates an interactive **Evidently AI** HTML report highlighting which features have drifted and by how much.

---

## 🔗 Related Project

This pipeline is the foundation for the **[Fraud Detection MLOps Platform](https://github.com/YOUR_USERNAME/fraud-monitoring-dashboard)** — an extended version that adds:

* SHAP explainability (`/explain` endpoint)
* Hot-swappable decision threshold (`/threshold` endpoint)
* Live Evidently drift dashboard (`/dashboard` endpoint)
* Prometheus metrics (`/metrics` endpoint)
* Redis-based transaction deduplication
* MLflow experiment tracking and model registry
* GitHub Actions automated retraining pipeline

**Live MLOps platform:** [https://fraud-metrics-dashboard.onrender.com](https://fraud-metrics-dashboard.onrender.com)
