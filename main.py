# main.py (Corrected Version)

# 1. Library Imports
import pandas as pd
import joblib
import time
import logging
import os # Make sure os is imported
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom engine functions
from engine.nba_engine import recommend_action
from engine.personalization_engine import generate_personalized_email

# --- CORRECTED: Set up a dedicated JSON logger to a writable directory ---
def setup_logger():
    logger = logging.getLogger('prediction_logger')
    logger.setLevel(logging.INFO)
    
    # Use the /tmp directory, which is writable in Hugging Face Spaces
    handler = logging.FileHandler('/tmp/prediction_logs.jsonl', mode='a')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger

prediction_logger = setup_logger()
# --- END of correction ---


# 2. Application Initialization
app = FastAPI(
    title="AI Customer Journey Optimizer API",
    description="An API to predict customer churn, recommend actions, and generate personalized messages.",
    version="1.0.0"
)


# 3. Loading Models and Data at Startup
@app.on_event("startup")
def load_resources():
    global churn_model, customer_data
    print("Loading resources: churn model and customer data...")
    
    churn_model = joblib.load('models/churn_model_pipeline.joblib')
    customer_data = pd.read_csv('data/crm_data.csv').set_index('CustomerID')
    
    # We no longer need to create the 'logs' directory
    
    print("Resources loaded successfully.")


# 4. Define Request and Response Models
class PredictionResponse(BaseModel):
    customer_id: str
    model_version: str
    churn_probability: float
    recommended_action: str
    personalized_email: str


# 5. Define API Endpoints
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the AI Customer Journey Optimizer API!"}


@app.post("/predict/{customer_id}", response_model=PredictionResponse)
def get_prediction(customer_id: str):
    if customer_id not in customer_data.index:
        raise HTTPException(status_code=404, detail="Customer ID not found.")
    
    customer_profile = customer_data.loc[customer_id]
    customer_df = pd.DataFrame([customer_profile])
    
    churn_prob = churn_model.predict_proba(customer_df)[0][1]
    action = recommend_action(customer_profile.to_dict())
    
    email = "No email generated (customer not considered at-risk)."
    if churn_prob > 0.5:
        email = generate_personalized_email(customer_profile.to_dict(), action)
    
    log_entry = {
        "timestamp": int(time.time()),
        "model_version": "1.0.0",
        "customer_id": customer_id,
        "features": {
            "Tenure": customer_profile.get("Tenure"),
            "UsageFrequency": customer_profile.get("UsageFrequency"),
            "SupportTickets": customer_profile.get("SupportTickets"),
            "MonthlyRevenue": customer_profile.get("MonthlyRevenue")
        },
        "prediction": {
            "churn_probability": round(churn_prob, 4),
            "recommended_action": action
        },
        "ground_truth_churn": None
    }
    prediction_logger.info(log_entry)

    return PredictionResponse(
        customer_id=customer_id,
        model_version="1.0.0",
        churn_probability=round(churn_prob, 4),
        recommended_action=action,
        personalized_email=email
    )