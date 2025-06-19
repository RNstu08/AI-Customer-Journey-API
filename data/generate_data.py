# data/generate_data.py

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime

# Initialize Faker for generating realistic fake data
fake = Faker()

# --- Configuration ---
NUM_CUSTOMERS = 10000
FILE_PATH = "data/crm_data.csv"

# --- Define possible values ---
SUBSCRIPTION_TIERS = ["Basic", "Standard", "Premium"]
LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]

# --- Generate Base Customer Data ---
print("Generating base customer data...")
data = {
    'CustomerID': [fake.uuid4() for _ in range(NUM_CUSTOMERS)],
    'Age': [random.randint(22, 65) for _ in range(NUM_CUSTOMERS)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(NUM_CUSTOMERS)],
    'Location': [random.choice(LOCATIONS) for _ in range(NUM_CUSTOMERS)],
    'SubscriptionTier': [random.choice(SUBSCRIPTION_TIERS) for _ in range(NUM_CUSTOMERS)],
    'Tenure': [random.randint(1, 60) for _ in range(NUM_CUSTOMERS)], # Tenure in months
}

df = pd.DataFrame(data)

# --- Engineer Correlated Features (to make the data realistic) ---
print("Engineering correlated features...")

# MonthlyRevenue is correlated with SubscriptionTier
def get_revenue(tier):
    if tier == 'Basic':
        return round(random.uniform(10, 30), 2)
    elif tier == 'Standard':
        return round(random.uniform(40, 70), 2)
    else: # Premium
        return round(random.uniform(80, 150), 2)

df['MonthlyRevenue'] = df['SubscriptionTier'].apply(get_revenue)

# UsageFrequency is correlated with Tenure (loyal customers use it more)
df['UsageFrequency'] = (df['Tenure'] * 1.5 + np.random.normal(0, 10, NUM_CUSTOMERS)).astype(int).clip(lower=1)

# SupportTickets are generally low, but higher for new customers
df['SupportTickets'] = (np.random.poisson(lam=3, size=NUM_CUSTOMERS) - (df['Tenure'] / 12)).astype(int).clip(lower=0)

# LastInteraction is random for now
df['LastInteraction'] = [random.randint(1, 90) for _ in range(NUM_CUSTOMERS)]


# --- Engineer the Target Variable: Churn ---
# This is the most important part. We create rules that define a "churn-prone" customer.
print("Engineering the target variable (Churn)...")

churn_probability = pd.Series(np.zeros(NUM_CUSTOMERS))

# Rule 1: High tenure, low usage is a big red flag
churn_probability += np.where((df['Tenure'] > 24) & (df['UsageFrequency'] < 20), 0.3, 0)

# Rule 2: High support tickets is a sign of issues
churn_probability += np.where(df['SupportTickets'] > 5, 0.2, 0)

# Rule 3: Low tenure and high support tickets is very bad
churn_probability += np.where((df['Tenure'] < 6) & (df['SupportTickets'] > 3), 0.4, 0)

# Rule 4: Long time since last interaction
churn_probability += np.where(df['LastInteraction'] > 60, 0.25, 0)

# Add some random noise
churn_probability += np.random.uniform(0, 0.1, NUM_CUSTOMERS)

# Apply a threshold to create the binary churn label
# We are aiming for an overall churn rate of about 15-20%
churn_threshold = churn_probability.quantile(0.85)
df['Churn'] = (churn_probability > churn_threshold).astype(int)


# --- Final Touches ---
print(f"Finalizing data and saving to {FILE_PATH}...")
# Ensure dtypes are correct
df['Age'] = df['Age'].astype('int16')
df['Tenure'] = df['Tenure'].astype('int16')
df['UsageFrequency'] = df['UsageFrequency'].astype('int16')
df['SupportTickets'] = df['SupportTickets'].astype('int16')
df['LastInteraction'] = df['LastInteraction'].astype('int16')
df['MonthlyRevenue'] = df['MonthlyRevenue'].astype('float32')
df['Churn'] = df['Churn'].astype('int8')

# Save to CSV
df.to_csv(FILE_PATH, index=False)

print("\n--- Data Generation Complete ---")
print(f"Data saved to: {FILE_PATH}")
print("\nChurn Rate in generated data:")
print(df['Churn'].value_counts(normalize=True))
print("\nSample of the data:")
print(df.head())