# Cell 3: Define Mistral-formatted Prompt Templates

PROMPT_TEMPLATES = {
    "Proactive Support Call": """
    [INST]
    You are a friendly and empathetic Customer Success Manager at ConnectSphere.
    Your goal is to proactively offer help to a customer who might be struggling, based on their recent activity.
    DO NOT sell anything. Your tone should be supportive and genuinely helpful.

    Customer Data:
    - Name: {Name}
    - Subscription Tier: {SubscriptionTier}
    - Recent Support Tickets: {SupportTickets}

    Task:
    Write a short, personal email (around 80 words) to {Name}.
    1. Start by acknowledging them as a valued customer.
    2. Mention that you noticed they've been in touch with support recently.
    3. Offer a direct, no-pressure 15-minute call to see if you can help them achieve their goals with the product.
    4. End with a friendly closing.
    [/INST]
    """,

    "20% Discount Offer": """
    [INST]
    You are a helpful marketing specialist at ConnectSphere.
    Your goal is to offer a special discount to a customer we've identified as being at-risk, framing it as a 'thank you' for their loyalty.
    Your tone should be appreciative and value-focused, not desperate.

    Customer Data:
    - Name: {Name}
    - Subscription Tier: {SubscriptionTier}
    - Months with us (Tenure): {Tenure}

    Task:
    Write a compelling email (around 100 words) to {Name}.
    1. Thank them for being a customer for {Tenure} months.
    2. Mention that you want to offer them a special 20% discount on their {SubscriptionTier} plan for the next 3 months as a token of appreciation.
    3. Briefly hint at a recent product update or benefit relevant to their tier.
    4. Provide a clear (but fictional) call to action to claim the offer.
    [/INST]
    """,

    "Send Educational Content": """
    [INST]
    You are a product expert at ConnectSphere.
    Your goal is to re-engage a new customer who seems inactive by showing them how to get more value from the product.
    Your tone should be encouraging and informative.

    Customer Data:
    - Name: {Name}
    - Subscription Tier: {SubscriptionTier}
    - Usage Frequency (logins last 30 days): {UsageFrequency}

    Task:
    Write a helpful email (around 120 words) to {Name}.
    1. Welcome them again to the {SubscriptionTier} plan.
    2. Acknowledge that getting started can be overwhelming.
    3. Based on their low usage, suggest a specific, high-impact feature they might have missed. (e.g., 'Automated Reporting' for Standard/Premium, 'Project Templates' for Basic). Be creative.
    4. Provide a link to a (fictional) short video tutorial or guide on that feature.
    5. End with an encouraging message, inviting them to explore.
    [/INST]
    """
}

# Cell 4: Create the Email Generation Function (using Hugging Face)
def generate_personalized_email(customer_data, action):
    """
    Generates a personalized email using the Hugging Face Inference API.

    Args:
        customer_data (dict): A dictionary with the customer's info.
        action (str): The next-best-action to take.

    Returns:
        str: The generated email content, or an error message.
    """
    if action not in PROMPT_TEMPLATES:
        return "Error: No prompt template found for the given action."

    # Add a fake name for the example
    if 'Name' not in customer_data:
        customer_data['Name'] = fake.first_name()

    prompt = PROMPT_TEMPLATES[action].format(**customer_data)

    try:
        print(f"--- Calling Hugging Face Inference API for action: {action} ---")
        # Use the client to generate text
        # Note: The first time you run this for a model, it might take longer as the model loads on the server.
        response = client.text_generation(
            prompt,
            max_new_tokens=250,  # Max length of the generated email
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        return response.strip()

    except Exception as e:
        return f"An error occurred with the Hugging Face API: {e}"

# We need a fake name generator for this example
from faker import Faker
fake = Faker()