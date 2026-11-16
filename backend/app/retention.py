
from typing import Any


def generate_retention_plan(
    customer: dict[str, Any],
    churn_probability: float,
) -> dict[str, Any]:
    """Generate transparent, rule-based retention suggestions."""

    probability = max(0.0, min(1.0, float(churn_probability)))

    if probability >= 0.70:
        risk = "High"
        priority = "Urgent review"
    elif probability >= 0.40:
        risk = "Medium"
        priority = "Follow up"
    else:
        risk = "Low"
        priority = "Routine monitoring"

    # Normalize feature names without requiring a fixed dataset schema.
    normalized = {
        str(key).strip().lower().replace("_", "").replace(" ", ""):
        value
        for key, value in customer.items()
    }

    actions = []

    contract = str(
        normalized.get("contract", "")
    ).strip().lower()

    if "month-to-month" in contract or contract == "monthly":
        actions.append({
            "action": "Review contract options",
            "reason": "The customer has a monthly contract.",
            "priority": "High",
        })

    tenure = normalized.get("tenure")

    try:
        if tenure is not None and float(tenure) < 12:
            actions.append({
                "action": "Offer onboarding support",
                "reason": "The customer has been with the service for less than a year.",
                "priority": "Medium",
            })
    except (ValueError, TypeError):
        pass

    payment = str(
        normalized.get("paymentmethod", "")
    ).lower()

    if "electronic check" in payment:
        actions.append({
            "action": "Review payment experience",
            "reason": "Check whether the customer needs help with payment options.",
            "priority": "Medium",
        })

    internet = str(
        normalized.get("internetservice", "")
    ).lower()

    if internet == "fiber optic":
        actions.append({
            "action": "Review service satisfaction",
            "reason": "Ask whether the customer is satisfied with service quality and pricing.",
            "priority": "Medium",
        })

    if not actions:
        actions.append({
            "action": "Conduct a customer satisfaction check-in",
            "reason": "Collect feedback to identify possible reasons for leaving.",
            "priority": "Medium",
        })

    if risk == "Low":
        actions = [
            {
                "action": "Maintain customer engagement",
                "reason": "The predicted churn risk is relatively low.",
                "priority": "Low",
            }
        ]

    return {
        "risk_segment": risk,
        "churn_probability_percent": round(probability * 100, 2),
        "priority": priority,
        "recommendations": actions,
        "disclaimer": (
            "Recommendations are suggestions based on available data. "
            "Validate customer needs before offering incentives."
        ),
    }