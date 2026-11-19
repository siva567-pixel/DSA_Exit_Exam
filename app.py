from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().with_name("shopping_model.pkl")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

FIELD_GROUPS = [
    {
        "title": "About you",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "min": 16, "max": 100},
            {
                "name": "Gender",
                "label": "Gender",
                "options": ["Female", "Male", "Prefer not to say"],
            },
            {
                "name": "Occupation_Status",
                "label": "Occupation",
                "options": [
                    "Student",
                    "Employed full-time",
                    "Employed part-time",
                    "Self-employed",
                    "Unemployed",
                ],
            },
            {
                "name": "Country_Region",
                "label": "Country or region",
                "options": [
                    "Brazil",
                    "Colombia",
                    "Egypt",
                    "Germany",
                    "India",
                    "Indonesia",
                    "Kenya",
                    "Mexico",
                    "Nigeria",
                    "Pakistan",
                    "Philippines",
                    "Poland",
                    "Slovakia",
                    "South Africa",
                    "Turkiye",
                    "United Kingdom",
                    "United States",
                    "Vietnam",
                ],
            },
            {
                "name": "Monthly_Income_Range",
                "label": "Monthly income",
                "options": ["Under $1,000", "$1,000-3,000", "$3,000-6,000", "$6,000+"],
            },
        ],
    },
    {
        "title": "Shopping habits",
        "fields": [
            {
                "name": "Shop_Most_Where",
                "label": "Where do you shop most?",
                "options": ["Online only", "In-store only", "Equal mix"],
            },
            {
                "name": "Online_Shopping_Frequency",
                "label": "How often do you shop online?",
                "options": ["Rarely", "Occasionally", "Often", "Always"],
            },
            {
                "name": "Preferred_Platform",
                "label": "Preferred shopping platform",
                "options": [
                    "Local e-commerce",
                    "Global marketplace (Amazon etc.)",
                    "Brand websites",
                    "Physical stores",
                ],
            },
            {
                "name": "Primary_Device",
                "label": "Primary shopping device",
                "options": ["Smartphone", "Laptop", "Tablet"],
            },
            {
                "name": "Top_Spending_Category",
                "label": "Top spending category",
                "options": [
                    "Clothing",
                    "Electronics",
                    "Groceries",
                    "Books",
                    "Beauty & personal care",
                    "miscellaneous",
                ],
            },
            {
                "name": "Review_Influence_Score",
                "label": "How much do reviews influence your purchases?",
                "type": "rating",
            },
            {
                "name": "Social_Ads_Influence_Score",
                "label": "How much do social media ads influence you?",
                "type": "rating",
            },
            {
                "name": "Price_Comparison_Frequency",
                "label": "How often do you compare prices?",
                "type": "rating",
            },
            {
                "name": "Discount_Importance_Score",
                "label": "How important are discounts to you?",
                "type": "rating",
            },
            {
                "name": "Follows_Brands_On_Social",
                "label": "Do you follow brands on social media?",
                "options": ["Yes", "No"],
            },
            {
                "name": "Likely_To_Buy_After_Ad_Score",
                "label": "How likely are you to buy after seeing an ad?",
                "type": "rating",
            },
            {
                "name": "Makes_Shopping_List",
                "label": "How often do you make a shopping list?",
                "options": ["Always", "Often", "Sometimes", "Never"],
            },
            {
                "name": "Impulse_Purchase_Frequency",
                "label": "How often do you make impulse purchases?",
                "type": "rating",
            },
            {
                "name": "Return_Frequency",
                "label": "How often do you return purchases?",
                "options": ["Never", "Often", "Sometimes"],
            },
            {
                "name": "Avg_Monthly_Spend_NonEssentials",
                "label": "Monthly non-essential spending",
                "options": ["Under $50", "$50-150", "$150-300", "$300-500", "$500+"],
            },
            {
                "name": "Preferred_Payment_Method",
                "label": "Preferred payment method",
                "options": ["Cash", "Credit Card", "Debit Card", "Digital wallet"],
            },
            {
                "name": "Uses_BNPL_Installments",
                "label": "Do you use buy-now-pay-later installments?",
                "options": ["No, never", "Yes, occasionally", "Yes, regularly"],
            },
            {
                "name": "Online_Shopping_Satisfaction_Score",
                "label": "How satisfied are you with online shopping?",
                "type": "rating",
            },
            {
                "name": "Trust_In_Online_Reviews_Score",
                "label": "How much do you trust online reviews?",
                "type": "rating",
            },
            {
                "name": "Regret_After_Purchase_Frequency",
                "label": "How often do you regret a purchase?",
                "type": "rating",
            },
        ],
    },
]

FORM_FIELDS = [
    field
    for group in FIELD_GROUPS
    for field in group["fields"]
]
FEATURE_COLUMNS = [field["name"] for field in FORM_FIELDS]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    submitted_data = {}

    if request.method == "POST":
        submitted_data = {
            field["name"]: request.form.get(field["name"], "").strip()
            for field in FORM_FIELDS
        }

        try:
            input_values = {}
            for field in FORM_FIELDS:
                name = field["name"]
                value = submitted_data[name]
                if not value:
                    raise ValueError("Please complete all fields before submitting.")

                if field.get("type") == "number":
                    numeric_value = float(value)
                    if not field["min"] <= numeric_value <= field["max"]:
                        raise ValueError(
                            f"{field['label']} must be between "
                            f"{field['min']} and {field['max']}."
                        )
                    input_values[name] = numeric_value
                elif field.get("type") == "rating":
                    numeric_value = int(value)
                    if not 1 <= numeric_value <= 5:
                        raise ValueError("Ratings must be from 1 to 5.")
                    input_values[name] = numeric_value
                else:
                    if value not in field["options"]:
                        raise ValueError(f"Select a valid option for {field['label']}.")
                    input_values[name] = value

            input_data = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
            prediction = model.predict(input_data)[0]
        except ValueError as exception:
            error = str(exception)

    return render_template(
        "predict.html",
        field_groups=FIELD_GROUPS,
        prediction=prediction,
        error=error,
        submitted_data=submitted_data,
    )


if __name__ == "__main__":
    app.run()
