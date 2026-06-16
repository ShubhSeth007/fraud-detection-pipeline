# High-Throughput Financial Fraud Detection Pipeline

An end-to-end, production-grade machine learning pipeline designed to detect fraudulent transactions under extreme class imbalance. This project moves beyond isolated Jupyter notebooks by embedding an optimized **XGBoost** classifier inside a containerized **FastAPI** application, complete with automated data drift monitoring using **Evidently AI**.

---

## 🛠️ Key Technical Features

* **Extreme Class Imbalance Handling:** Modeled on a highly skewed distribution (99.5% legitimate, 0.5% fraud) utilizing stratified cross-validation and tuned `scale_pos_weight` parameters to prevent majority-class bias.
* **Advanced Feature Engineering:** Simulates transaction frequency, time-based components (Hour of Day), behavioral aggregations (deviation from card transaction means), and rolling time deltas to capture complex fraud patterns.
* **Rigorous Hyperparameter Optimization:** Uses **Optuna** to optimize hyperparameter spaces over multiple folds, targeting **Precision-Recall AUC (PR-AUC)** rather than standard accuracy, which fails on imbalanced data.
* **Production Serving Engine:** Built an asynchronous **FastAPI** wrapper with rigid structural data validation enforced via **Pydantic** models.
* **Enterprise Containerization:** Fully containerized via a lean, multi-worker **Docker** configuration optimized for high-throughput cloud environments like Render.
* **MLOps Observability:** Integrates **Evidently AI** statistical profiling suites to monitor incoming feature shifts and data drift against training baselines.

---

## 📂 Project Repository Structure

```text
├── app.py                  # Asynchronous FastAPI production serving engine
├── train.py                # Kaggle-optimized model training & Optuna tuning script
├── monitor_drift.py        # Evidently AI data drift validation suite
├── Dockerfile              # Lean containerization config optimized for cloud deployment
├── requirements.txt        # Rigid environment dependencies layout
├── fraud_model.joblib      # Serialized production XGBoost model artifact (Generated via Kaggle)
└── feature_columns.joblib  # Serialized model-specific feature schema alignment
```

---

## 📊 Model Performance

Standard accuracy is a deceptive metric for fraud detection; a naive model guessing "Never Fraud" achieves 99.5% accuracy but catches 0% of fraud.

This model explicitly optimizes for **PR-AUC (Average Precision)**:

| Metric | Score |
|---|---|
| Baseline Random Guessing | 0.005 (based on positive class prevalence) |
| Optimized Model PR-AUC | **0.2638** |

Our hyperparameter-tuned model performs over **52 times better than random chance**, heavily suppressing false positives to preserve user experience while accurately catching malicious actors.

---

## ⚙️ Local Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### 2. Run Locally via Docker

Build and run the containerized application instantly without worrying about local Python dependency mismatches:

```bash
# Build the Docker image
docker build -t fraud-detection-service:v1 .

# Spin up the container locally on port 8000
docker run -p 8000:10000 -e PORT=10000 fraud-detection-service:v1
```

### 3. Verify Local Deployment

Open your browser and navigate to `http://localhost:8000/docs` to interact with the live automated Swagger documentation interface.

---

## 🔌 API Documentation & Endpoints

### `GET /health`

Validates model readiness and infrastructure system health.

### `POST /predict`

Scores incoming real-time transactions for fraud risk.

**Sample Request Payload:**

```json
{
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
}
```

**Sample API Response:**

```json
{
  "fraud_probability": 0.0241,
  "is_fraud": 0,
  "status": "APPROVED"
}
```

> **Note:** Any transaction crossing a high-risk probability threshold is automatically flagged as `FLAGGED_FOR_REVIEW` for human triage.

---

## 📉 Production Observability (Data Drift)

To evaluate if production input logs have diverged significantly from the initial baseline training sets, execute the automated drift tracking engine:

```bash
python monitor_drift.py
```
