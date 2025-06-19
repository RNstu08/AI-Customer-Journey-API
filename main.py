# main.py

# 1. Library Imports
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom engine functions
from engine.nba_engine import recommend_action
from engine.personalization_engine import generate_personalized_email


# 2. Application Initialization
app = FastAPI(
    title="AI Customer Journey Optimizer API",
    description="An API to predict customer churn, recommend actions, and generate personalized messages.",
    version="1.0.0"
)


# 3. Loading Models and Data at Startup
# This is done once when the API server starts to avoid reloading on every request.
@app.on_event("startup")
def load_resources():
    global churn_model, customer_data
    print("Loading resources: churn model and customer data...")
    
    # Load the churn prediction pipeline
    churn_model = joblib.load('models/churn_model_pipeline.joblib')
    
    # Load the customer data (in a real app, this would be a database connection)
    customer_data = pd.read_csv('data/crm_data.csv').set_index('CustomerID')
    
    print("Resources loaded successfully.")


# 4. Define Request and Response Models
# Pydantic models for type validation and clear API documentation
class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    recommended_action: str
    personalized_email: str


# 5. Define API Endpoints
@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the AI Customer Journey Optimizer API!"}


@app.post("/predict/{customer_id}", response_model=PredictionResponse)
def get_prediction(customer_id: str):
    """
    Generates a full prediction for a given customer ID.
    - Predicts churn probability.
    - Recommends the next-best-action.
    - Generates a personalized email for that action.
    """
    # --- 1. Fetch Customer Data ---
    if customer_id not in customer_data.index:
        raise HTTPException(status_code=404, detail="Customer ID not found.")
    
    customer_profile = customer_data.loc[customer_id]
    
    # --- 2. Run Churn Prediction (Phase 3) ---
    # The model pipeline expects a DataFrame, so we create one from our Series
    customer_df = pd.DataFrame([customer_profile])
    # Use predict_proba to get the probability of the positive class (Churn=1)
    churn_prob = churn_model.predict_proba(customer_df)[0][1]

    # --- 3. Get Next-Best-Action (Phase 4) ---
    # Our NBA engine expects a dictionary
    action = recommend_action(customer_profile.to_dict())

    # --- 4. Generate Personalized Email (Phase 5) ---
    # Only generate an email if the customer is at risk (e.g., prob > 50%)
    email = "No email generated (customer not considered at-risk)."
    if churn_prob > 0.5:
        email = generate_personalized_email(customer_profile.to_dict(), action)

    # --- 5. Package and Return Response ---
    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=round(churn_prob, 4),
        recommended_action=action,
        personalized_email=email
    )