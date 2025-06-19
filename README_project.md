Here's a well-structured and detailed Markdown README for your AI-Powered Customer Journey Optimization project, based on the information you provided:

---

# AI-Powered Customer Journey Optimization (CRM-Integrated)

## 1. Project Overview

This project is a comprehensive, end-to-end simulation of an enterprise-grade AI system designed to proactively reduce customer churn and increase customer lifetime value (CLTV). The system integrates with a simulated CRM to analyze customer data, predict churn likelihood, recommend the optimal retention strategy (Next-Best-Action), and generate hyper-personalized outreach emails using Generative AI.

The entire project is built following industry best practices for MLOps, including model versioning, a central model registry, CI/CD-ready deployment via containerization, and a final A/B test to prove the system's business value and ROI.

**Business Goal:** To move from reactive customer service to a proactive, data-driven, and personalized customer retention strategy, thereby reducing churn and increasing revenue.

## 2. Core Features

* **Churn Prediction:** A machine learning model (Gradient Boosting) trained on customer data to predict the probability of a customer churning.
* **Next-Best-Action (NBA) Engine:** A rule-based system that analyzes a customer's profile and recommends the most effective retention action (e.g., Proactive Support Call, 20% Discount Offer).
* **Hyper-Personalized Content Generation:** Leverages a powerful open-source Large Language Model (Mistral-7B via Hugging Face) to craft unique, context-aware outreach emails for the recommended action.
* **Enterprise-Grade MLOps Architecture:**
    * **MLflow Model Registry:** Manages the lifecycle of all trained models, including versioning, staging (Staging vs. Production), and artifact storage.
    * **Decoupled Artifact Storage:** Uses AWS S3 as the backend artifact store, separating large model files from the application codebase.
    * **Live API Deployment:** The entire system is served via a FastAPI application, containerized with Docker, and deployed on Hugging Face Spaces for public access.
    * **A/B Testing Framework:** The live API includes logic to run a controlled experiment, splitting at-risk customers into a 'Control' group and a 'Treatment' group to scientifically measure the impact of the AI interventions.

## 3. Technology Stack & Tools

| Category             | Technology / Tool             | Purpose                                                            |
| :------------------- | :---------------------------- | :----------------------------------------------------------------- |
| **Backend** | Python, FastAPI               | Building the robust, high-performance API server.                  |
| **ML / Data Science**| Pandas, Scikit-learn, Jupyter | Data manipulation, model training, and exploratory analysis.       |
| **Generative AI** | Hugging Face Inference API (Mistral-7B) | Generating personalized email content.                           |
| **MLOps & Deployment**| Docker, Git, Git LFS         | Containerization and version control for a portable application.   |
|                      | MLflow                        | Experiment tracking and central Model Registry.                    |
|                      | AWS S3 & IAM                  | Secure, scalable cloud storage for model artifacts.                |
|                      | Hugging Face Spaces           | Cloud platform for hosting the live Dockerized application.        |
| **Server** | Uvicorn                       | High-performance ASGI server for running the FastAPI app.          |

## 4. High-Level Architecture

The system is designed with a decoupled, professional architecture that separates concerns for scalability and security.

```
Client (e.g., CRM)
    ↓
FastAPI on Hugging Face Spaces (Application Host)
    ↓ (On Startup)
Authenticates via IAM Keys
    ↓
Queries MLflow Model Registry
    ↓
MLflow provides S3 path for Production model
    ↓
Application loads model from AWS S3 (Artifact Store)
    ↓ (On Request)
Generates Prediction & Recommendation
    ↓
Sends Response to Client
```

## 5. Project Structure

```
ai-customer-journey/
│
├── .git/                 # Git version control directory
├── .gitignore            # Specifies intentionally untracked files to ignore
├── .gitattributes        # Used by Git LFS to track large files
│
├── data/                 # For all raw and generated data
│   ├── crm_data.csv
│   ├── action_outcomes.csv
│   ├── generate_data.py
│   └── generate_action_data.py
│
├── engine/               # Core application logic modules
│   ├── nba_engine.py
│   └── personalization_engine.py
│
├── models/               # (Legacy) For storing locally saved models
│   └── churn_model_pipeline.joblib # Now managed by MLflow/S3
│
├── mlruns/               # Local directory for MLflow experiment metadata (ignored by LFS for binary artifacts)
│
├── venv/                 # Python virtual environment (ignored by Git)
│
├── 01_EDA.ipynb          # Notebook for Exploratory Data Analysis
├── 02_Churn_Model_Training.ipynb # Notebook for training the model and logging to MLflow
├── 03_Next_Best_Action.ipynb # Notebook for developing the NBA engine
├── 04_GenAI_Personalization.ipynb # Notebook for developing GenAI email generation
├── 05_Monitoring_Dashboard.ipynb # Notebook for analyzing production logs (conceptual)
├── 06_AB_Test_Analysis.ipynb # Notebook for analyzing the final A/B test results
│
├── main.py               # The main FastAPI application file
├── Dockerfile            # Recipe for building the production Docker container
├── .dockerignore         # Specifies files to exclude from the container
├── requirements.txt      # List of all Python dependencies for the project
└── README.md             # This file!
```

## 6. Local Setup and Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* **Python 3.9+**
* **Git** and **Git LFS** (`git lfs install` after Git installation)
* **Docker Desktop** (for Windows, ensure WSL 2 is enabled and healthy)
* An **AWS account** with an S3 bucket and IAM user credentials (Access Key ID & Secret Access Key) having permissions for S3.
* A **Hugging Face account** with an Access Token (with "write" role).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ai-customer-journey
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables (Crucial for Security):**
    Create a file named `.env` in the root of the project directory. This file should **NOT** be committed to Git. Ensure `.env` is listed in your `.gitignore` file. Add your secret credentials to the `.env` file like this:
    ```ini
    # .env file
    AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID_HERE"
    AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY_HERE"
    HF_TOKEN="YOUR_HUGGING_FACE_TOKEN_HERE"
    ```

## 7. How to Run the Project

The project is designed to be run in sequence.

1.  **Generate Data (Optional):**
    If you don't have the `crm_data.csv` and `action_outcomes.csv` files, run the generation scripts:
    ```bash
    python data/generate_data.py
    python data/generate_action_data.py
    ```

2.  **Train and Register the Model:**
    Open and run the cells in `02_Churn_Model_Training.ipynb`.
    This notebook will train the model, use the credentials from your `.env` file to log the model artifact to your S3 bucket, and register it in the MLflow Model Registry.

3.  **View and Promote the Model in MLflow:**
    In your terminal (from the project root, with `venv` active), start the MLflow UI:
    ```bash
    mlflow ui
    ```
    Navigate to `http://127.0.0.1:5000` (or the URL provided) in your browser.
    Go to the "Models" tab, find `churn-predictor`, and promote the latest version to the "Production" stage.

4.  **Run the API Server Locally:**
    Once a model is in "Production" (and your `main.py` is configured to load from MLflow registry), you can run the web server:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`. Access the interactive documentation at `http://127.0.0.1:8000/docs`.

5.  **Analyze the A/B Test Results:**
    Open and run the cells in `06_AB_Test_Analysis.ipynb`.
    This notebook will call your live deployed API (or your local one if you change the URL in the notebook's configuration) to simulate the experiment and then analyze the results to calculate the ROI.

## 8. Deployment

The application is deployed as a Docker container on Hugging Face Spaces.

* The `Dockerfile` defines the container environment, installs dependencies, and specifies the command to run the Uvicorn server.
* The `Hugging Face Space` is configured to use Docker as its SDK.
* Secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`) are securely stored in the Space's settings and are injected as environment variables at runtime. This ensures they are never exposed in the codebase.
* The application dynamically loads the production-stage model from the S3-backed MLflow registry on startup.

## 9. Future Improvements

* **CI/CD Pipeline:** Implement GitHub Actions to automatically test and deploy the application to Hugging Face Spaces on every push to the `main` branch.
* **Real-time Event Processing:** Replace the static `crm_data.csv` with a connection to a real database (e.g., PostgreSQL) and use a message queue like Kafka to process customer events in real-time.
* **Advanced Monitoring:** Integrate a dedicated logging service (like Datadog or Grafana Loki) to create dashboards for monitoring model drift, API latency, and error rates.
* **More Complex Models:** Experiment with more advanced churn prediction models (e.g., LSTMs for sequence-based behavior) or a Reinforcement Learning approach for the Next-Best-Action engine.

---