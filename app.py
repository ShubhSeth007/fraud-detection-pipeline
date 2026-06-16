import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="High-Throughput Fraud Detection API",
    description="Production endpoint for scoring real-time fraud probability.",
    version="1.0.0"
)

MODEL = None
FEATURES = None

@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURES
    if not os.path.exists("fraud_model.joblib") or not os.path.exists("feature_columns.joblib"):
        raise RuntimeError("Model artifacts missing in the working directory.")
    MODEL = joblib.load("fraud_model.joblib")
    FEATURES = joblib.load("feature_columns.joblib")
    print("Model and features mapped to memory successfully.")

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., example=150.50)
    card1: float = Field(..., example=1234.0)
    card2: float = Field(..., example=567.0)
    addr1: float = Field(..., example=89.0)
    dist1: float = Field(..., example=12.5)
    C1: float = Field(..., example=1.0)
    C2: float = Field(..., example=2.0)
    C3: float = Field(..., example=0.0)
    C4: float = Field(..., example=0.0)
    C5: float = Field(..., example=1.0)
    C6: float = Field(..., example=1.0)
    C7: float = Field(..., example=0.0)
    C8: float = Field(..., example=0.0)
    C9: float = Field(..., example=1.0)
    C10: float = Field(..., example=0.0)
    TransactionHour: int = Field(..., example=14)
    TransactionAmt_to_mean_card1: float = Field(..., example=1.12)
    TransactionAmt_to_std_card1: float = Field(..., example=0.45)
    time_delta_card1: float = Field(..., example=360.0)

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: int
    status: str

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionInput):
    if MODEL is None or FEATURES is None:
        raise HTTPException(status_code=500, detail="Inference engine uninitialized.")
    
    try:
        input_dict = payload.dict()
        input_df = pd.DataFrame([input_dict])[FEATURES]
        
        prob = float(MODEL.predict_proba(input_df)[0][1])
        prediction = int(MODEL.predict(input_df)[0])
        status = "FLAGGED_FOR_REVIEW" if prob > 0.5 else "APPROVED"
        
        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=prediction,
            status=status
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}
