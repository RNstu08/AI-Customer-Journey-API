# Phase 4: Next-Best-Action (NBA) Recommendation Engine.

# We've successfully built a model to predict who is at risk of churning. 
# Now, we address the crucial follow-up question: What is the most effective action we can take to retain them?

# An NBA engine moves us from being reactive to being proactive. 
# Instead of treating all at-risk customers the same, we'll recommend a specific, targeted action tailored to their unique situation.

# The Challenge: The Lack of Historical Action Data
# In a real-world scenario, we would need historical data from past retention campaigns. 
# We'd need to know which customers received which offers (e.g., a discount, a support call) and whether those offers were successful.
# Since we don't have this data, we will simulate it based on logical business assumptions. 
# This allows us to build and test the entire system end-to-end.


# Step 4.1: Defining Our "Actions" and Simulating Outcomes
# First, let's define the possible retention actions our company, "ConnectSphere," can take:

# 20% Discount Offer: A direct financial incentive.
# Proactive Support Call: A customer success manager reaches out to help.
# Send Educational Content: An email with tutorials on how to better use the product.
# No Action: The control group. We do nothing and observe the outcome.
# Now, we'll create a script to generate a new data file, action_outcomes.csv. 
# This script will assign a random historical action to each customer from our original dataset and then simulate whether that action was successful (ChurnPrevented).

# The success of an action will depend on the customer's profile, based on these logical rules:

# Discounts work best on customers who are sensitive to price but are otherwise engaged.
# Support calls are most effective for customers who are actively struggling (high support tickets).
# Educational content is best for new customers who may not yet see the product's full value (low tenure, low usage).

import pandas as pd
import numpy as np

print("Loading existing CRM data...")
df = pd.read_csv('data/crm_data.csv')

# --- Configuration ---
ACTIONS = ["20% Discount Offer", "Proactive Support Call", "Send Educational Content", "No Action"]
FILE_PATH = "data/action_outcomes.csv"

# --- Simulate Historical Actions ---
print("Simulating historical actions taken for each customer...")
# Assign a random action to each customer
df['ActionTaken'] = np.random.choice(ACTIONS, size=len(df))

# --- Simulate the Outcome (ChurnPrevented) based on rules ---
print("Simulating action outcomes based on business logic...")
# Start with a baseline success probability for each action
baseline_success = {
    "20% Discount Offer": 0.30,      # 30% baseline success
    "Proactive Support Call": 0.25,  # 25% baseline success
    "Send Educational Content": 0.20, # 20% baseline success
    "No Action": 0.02                # 2% "self-cure" rate
}

# Calculate success probability for each row
def calculate_success_prob(row):
    action = row['ActionTaken']
    prob = baseline_success[action]

    # Rule 1: Discounts are more effective for high-revenue customers
    if action == "20% Discount Offer" and row['MonthlyRevenue'] > 75:
        prob += 0.30 # Big boost

    # Rule 2: Support calls are very effective for those with many tickets
    if action == "Proactive Support Call" and row['SupportTickets'] > 4:
        prob += 0.40 # Big boost

    # Rule 3: Education is effective for new, inactive users
    if action == "Send Educational Content" and row['Tenure'] < 12 and row['UsageFrequency'] < 20:
        prob += 0.35 # Big boost

    # Rule 4: Discounts are less effective on brand new customers
    if action == "20% Discount Offer" and row['Tenure'] < 6:
        prob -= 0.15

    # Rule 5: If churn was predicted, there's a chance to prevent it. Otherwise, no.
    if row['Churn'] == 0:
        return 0 # Cannot prevent churn that wasn't going to happen

    return min(max(prob, 0), 1) # Ensure probability is between 0 and 1

# Apply the logic
success_probabilities = df.apply(calculate_success_prob, axis=1)

# Determine the final outcome based on the probability
df['ChurnPrevented'] = (np.random.rand(len(df)) < success_probabilities).astype(int)

# --- Finalize and Save ---
print(f"Saving simulated action data to {FILE_PATH}...")
# We only need a subset of columns for this analysis
action_df = df[['CustomerID', 'Tenure', 'MonthlyRevenue', 'UsageFrequency', 'SupportTickets', 'ActionTaken', 'ChurnPrevented']]
action_df.to_csv(FILE_PATH, index=False)


print("\n--- Action Data Generation Complete ---")
print(f"Data saved to: {FILE_PATH}")
print("\nSuccess rate of each action (on customers who were predicted to churn):")
# Filter for customers who were originally going to churn to get meaningful stats
print(df[df['Churn'] == 1].groupby('ActionTaken')['ChurnPrevented'].mean())
print("\nSample of the new data:")
print(action_df.head())