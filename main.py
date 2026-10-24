import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from features import compute_features  
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    variant = os.getenv("MODEL_VARIANT", "no_hour").lower()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if variant == "hour":
        model_path = os.path.join(base_dir, "mnq_lightgbm_model_hour.pkl")
        cols_path = os.path.join(base_dir, "mnq_features_cols.pkl")
    else:
        model_path = os.path.join(base_dir, "mnq_lightgbm_model_no_hour.pkl")
        cols_path = os.path.join(base_dir, "mnq_features_cols_no_hour.pkl")
        
    print(f"Attempting to load model from: {model_path}")
    print(f"Attempting to load columns from: {cols_path}")
    
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        raise FileNotFoundError(f"Required artifact files not found for variant '{variant}' in directory '{base_dir}'.")
        
    ml_models["model"] = joblib.load(model_path)
    ml_models["feature_cols"] = joblib.load(cols_path)
    print(f"[{variant.upper()}] Model and {len(ml_models['feature_cols'])} feature columns loaded successfully into memory.")
    
    yield
    
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class OHLCVRequest(BaseModel):
    history: list[dict]

@app.post("/predict")
def predict(payload: OHLCVRequest):
    try:
        df = pd.DataFrame(payload.history)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }
        df.rename(columns=rename_map, inplace=True)
        df_features = compute_features(df)
        
        feature_cols = ml_models["feature_cols"]
        model = ml_models["model"]
        
        X_latest = df_features[feature_cols].tail(1)
        prediction = int(model.predict(X_latest)[0])
        probability = float(model.predict_proba(X_latest)[0][1]) if hasattr(model, "predict_proba") else 0.0
        
        signal = "BUY" if prediction == 1 else "SELL"
        
        return {
            "signal": signal,
            "probability": probability,
            "variant": os.getenv("MODEL_VARIANT", "no_hour")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)