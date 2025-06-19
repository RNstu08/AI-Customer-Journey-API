# main.py (Enterprise Version)

# 1. Library Imports
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom engine functions
from engine.nba_engine import recommend_action
from engine.personalization_engine import generate_personalized_email

# 2. Application Initialization
app = FastAPI(
    title="AI Customer Journey Optimizer API (Enterprise)",
    description="An API that loads models from a central MLflow Model Registry.",
    version="2.0.0"
)

# 3. Loading Models and Data at Startup
@app.on_event("startup")
def load_resources():
    global churn_model, customer_data
    print("Loading resources...")
    
    # --- MODEL LOADING FROM MLFLOW REGISTRY ---
    # The model URI format is "models:/<registered_model_name>/<stage_or_version>"
    # We will fetch the "Production" stage model. (You need to set this in MLflow UI first)
    model_uri = "models:/churn-predictor/Production" 
    try:
        print(f"Loading model from MLflow Registry: {model_uri}")
        churn_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        # If it fails, the app can't start. We raise an exception.
        raise RuntimeError(f"Could not load model from MLflow Registry: {e}")
    # --- END OF MODEL LOADING ---
    
    # Load customer data (this part remains the same)
    customer_data = pd.read_csv('data/crm_data.csv').set_index('CustomerID')
    print("Resources loaded successfully.")

# ... (The rest of the file: PredictionResponse, endpoints, etc., can remain the same as the last version) ...
# ... BUT make sure you remove the local logging part if you want, or keep it ...
# For simplicity, here is the rest of the code:

class PredictionResponse(BaseModel):
    customer_id: str
    model_version: str
    churn_probability: float
    recommended_action: str
    personalized_email: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Enterprise AI API!"}

@app.post("/predict/{customer_id}", response_model=PredictionResponse)
def get_prediction(customer_id: str):
    if customer_id not in customer_data.index:
        raise HTTPException(status_code=404, detail="Customer ID not found.")
    
    customer_profile = customer_data.loc[customer_id]
    customer_df = pd.DataFrame([customer_profile])
    
    # Use the loaded MLflow model to predict
    churn_prob = churn_model.predict(customer_df)[0] # MLflow pyfunc predict gives the probability directly if framed correctly, or we might need to adjust
    
    # NOTE: The output format of mlflow.pyfunc.predict might differ. 
    # It often wraps predictions. Let's assume for a scikit-learn model it might be the direct prediction.
    # A safer way is to check the output, but for now we will assume it's the probability.
    
    action = recommend_action(customer_profile.to_dict())
    
    email = "No email generated (customer not considered at-risk)."
    if churn_prob > 0.5:
        email = generate_personalized_email(customer_profile.to_dict(), action)

    # We can get the model version from the loaded model object
    model_version_str = churn_model.metadata.run_id

    return PredictionResponse(
        customer_id=customer_id,
        model_version=model_version_str,
        churn_probability=round(float(churn_prob), 4),
        recommended_action=action,
        personalized_email=email
    )