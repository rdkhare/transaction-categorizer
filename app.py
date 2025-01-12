import re
from flask import Flask, jsonify, request
from transformers import pipeline


app = Flask(__name__)

# Predefined categories and subcategories
categories = {
    'Auto': ['Gas', 'Maintenance', 'Upgrades', 'Other_Auto'],
    'Baby': ['Diapers', 'Formula', 'Clothes', 'Toys', 'Other_Baby'],
    'Clothes': ['Clothes', 'Shoes', 'Jewelry', 'Bags_Accessories'],
    'Entertainment': ['Sports_Outdoors', 'Movies_TV', 'DateNights', 'Arts_Crafts', 'Books', 'Games', 'Guns', 'E_Other'],
    'Electronics': ['Accessories', 'Computer', 'TV', 'Camera', 'Phone', 'Tablet_Watch', 'Gaming', 'Electronics_misc'],
    'Food': ['Groceries', 'FastFood_Restaurants'],
    'Home': ['Maintenance', 'Furniture_Appliances', 'Hygiene', 'Gym', 'Home_Essentials', 'Kitchen', 'Decor', 'Security', 'Yard_Garden', 'Tools'],
    'Medical': ['Health_Wellness'],
    'Kids': ['K_Toys'],
    'Personal_Care': ['Hair', 'Makeup_Nails', 'Beauty', 'Massage', 'Vitamins_Supplements', 'PC_Other'],
    'Pets': ['Pet_Food', 'Pet_Toys', 'Pet_Med', 'Pet_Grooming', 'Pet_Other'],
    'Subscriptions_Memberships': ['Entertainment', 'Gym', 'Sub_Other'],
    'Travel': ['Hotels', 'Flights', 'Car_Rental', 'Activities']
}

# Helper function to preprocess the description
def preprocess_description(description):
    # Lowercase and remove special characters
    return re.sub(r"[^a-zA-Z0-9\s]", "", description.lower())

# Load Hugging Face zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define your categories as labels
main_categories = list(categories.keys())  # Use your predefined categories

# Enhanced categorization logic using NLP
def categorize_transaction(description):
    description = preprocess_description(description)
    result = classifier(description, main_categories)  # Classify description
    category = result["labels"][0]  # Get the highest-scoring category
    return category


# Categorization endpoint
@app.route("/categorize", methods=["POST"])
def categorize():
    data = request.get_json()
    description = data.get("description", "")

    if not description:
        return jsonify({"error": "Transaction description is required"}), 400

    category = categorize_transaction(description)
    return jsonify({"description": description, "category": category}), 200

# Health Check Route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "Transaction Categorizer API is running"}), 200

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
