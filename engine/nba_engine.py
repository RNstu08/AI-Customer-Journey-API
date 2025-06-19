# engine/nba_engine.py

# This dictionary holds the "learned" rules from our analysis.
# In a real system, this could be loaded from a config file or a database.
NBA_RULES = {
    'Struggling User': 'Proactive Support Call',
    'New and Inactive': 'Send Educational Content',
    'High-Value, Low-Engagement': '20% Discount Offer',
    'General At-Risk': '20% Discount Offer' # From our analysis, discount was best for the general group
}

def get_customer_segment(customer_data):
    """Assigns a customer to a pre-defined segment."""
    if customer_data.get('SupportTickets', 0) > 4:
        return 'Struggling User'
    elif customer_data.get('Tenure', 0) < 12 and customer_data.get('UsageFrequency', 0) < 20:
        return 'New and Inactive'
    elif customer_data.get('MonthlyRevenue', 0) > 75 and customer_data.get('UsageFrequency', 0) < 40:
        return 'High-Value, Low-Engagement'
    else:
        return 'General At-Risk'


def recommend_action(customer_data):
    """
    Recommends the next best action for a single customer.

    Args:
        customer_data (dict): A dictionary containing a customer's data.
                              Needs keys: 'SupportTickets', 'Tenure',
                              'UsageFrequency', 'MonthlyRevenue'.
    Returns:
        str: The recommended action.
    """
    # 1. Determine the customer's segment
    segment = get_customer_segment(customer_data)

    # 2. Return the best action for that segment
    return NBA_RULES.get(segment, "No Action")