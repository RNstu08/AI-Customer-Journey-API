# main.py (Final Version with A/B Testing)

# 1. Library Imports
import pandas as pd
import mlflow
import time
import logging
import os
import random # Import the random module
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.nba_engine import recommend_action
from engine.personalization_engine import generate_personalized_email

# Logger setup (remains the same)
def setup_logger():
    logger = logging.getLogger('prediction_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('/tmp/prediction_logs.jsonl', mode='a')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
prediction_logger = setup_logger()

# App Initialization
app = FastAPI(
    title="AI Customer Journey Optimizer API (with A/B Testing)",
    description="An API that serves AI recommendations and runs an A/B test to measure impact.",
    version="3.0.0"
)

# Resource Loading (remains the same)
@app.on_event("startup")
def load_resources():
    global churn_model, customer_data
    print("Loading resources...")
    model_uri = "models:/churn-predictor/Production" 
    try:
        print(f"Loading model from MLflow Registry: {model_uri}")
        churn_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Could not load model from MLflow Registry: {e}")
    customer_data = pd.read_csv('data/crm_data.csv').set_index('CustomerID')
    print("Resources loaded successfully.")

# --- UPDATED: Response Model with Experiment Info ---
class PredictionResponse(BaseModel):
    customer_id: str
    model_version: str
    churn_probability: float
    experiment_group: str # 'A' (Control), 'B' (Treatment), or 'N/A'
    action_taken: str
    personalized_email: str | None = None # Email is now optional


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the A/B Testing AI API!"}

@app.post("/predict/{customer_id}", response_model=PredictionResponse)
def get_prediction(customer_id: str):
    if customer_id not in customer_data.index:
        raise HTTPException(status_code=404, detail="Customer ID not found.")
    
    customer_profile = customer_data.loc[customer_id]
    customer_df = pd.DataFrame([customer_profile])
    
    churn_prob = churn_model.predict(customer_df)[0]
    model_version_str = churn_model.metadata.run_id

    # --- NEW: A/B Test Logic ---
    experiment_group = 'N/A' # Default for customers not at-risk
    action = 'N/A'
    email = None
    
    # Define a churn threshold for who enters the experiment
    CHURN_THRESHOLD = 0.5 

    if churn_prob >= CHURN_THRESHOLD:
        # This customer is at-risk and will be part of our experiment
        if random.random() < 0.5:
            # Group A (Control): 50% chance. Do nothing.
            experiment_group = 'A'
            action = 'No Action (Control Group)'
        else:
            # Group B (Treatment): 50% chance. Apply AI recommendation.
            experiment_group = 'B'
            action = recommend_action(customer_profile.to_dict())
            email = generate_personalized_email(customer_profile.to_dict(), action)
    else:
        # Customer is not at risk, not part of the experiment
        action = 'No Action (Not At-Risk)'
    # --- END of A/B Test Logic ---

    # Expanded logging to include experiment group
    log_entry = {
        "timestamp": int(time.time()),
        "model_version": model_version_str,
        "customer_id": customer_id,
        "features": customer_profile.to_dict(),
        "prediction": {"churn_probability": round(float(churn_prob), 4)},
        "experiment": {
            "group": experiment_group,
            "action_taken": action
        },
        "ground_truth_churn": None
    }
    prediction_logger.info(log_entry)

    return PredictionResponse(
        customer_id=customer_id,
        model_version=model_version_str,
        churn_probability=round(float(churn_prob), 4),
        experiment_group=experiment_group,
        action_taken=action,
        personalized_email=email
    )