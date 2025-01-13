import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report

# Load the saved LabelEncoder
label_encoder = joblib.load("models/saved_model/label_encoder.pkl")

# Load the trained model and tokenizer
model_path = "models/saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load the validation dataset
val_df = pd.read_csv("data/validation.csv")

# Prepare inputs and true labels
true_labels = val_df["category"].tolist()
predicted_labels = []

# Loop through the validation dataset and make predictions
for _, row in val_df.iterrows():
    description = row["description"]

    # Tokenize the input description
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Decode the predicted class to the original category
    predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    predicted_labels.append(predicted_category)

# Evaluate model performance
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Validation Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))
