import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

train_df = load_data("data/train.csv")
val_df = load_data("data/validation.csv")

# Encode Labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["category"])
val_df["label"] = label_encoder.transform(val_df["category"])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["description"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Training Arguments
training_args = TrainingArguments(
    output_dir="models/saved_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Keep smaller batch size
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("models/saved_model")
    tokenizer.save_pretrained("models/saved_model")
