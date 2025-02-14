import torch
import json
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your dataset
test = datasets.load_from_disk("data/test_dataset")

# Define model checkpoint
model_name = "data/results/checkpoint-76000"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Function to run prediction on a batch
@torch.inference_mode()
def predict_batch(batch):
    # Tokenize texts in the batch
    inputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Get predictions (argmax)
    pred_ids = logits.argmax(dim=-1)
    # Convert prediction ids to labels using the model's configuration
    pred_labels = [model.config.id2label[p.item()] for p in pred_ids]
    return {"predicted_label": pred_labels}

# Define your batch size
batch_size = 1024

# Use map to apply the predict_batch function over the dataset in batches
predictions = test.map(predict_batch, batched=True, batch_size=batch_size)

# Combine predictions with ground truth labels
results = {
    "predictions": predictions["predicted_label"],
    "expected": predictions["lang"]
}

# Save the final predictions to a JSON file
with open("data/predictions/final_predictions.json", "w") as f:
    json.dump(results, f)

print("Prediction complete. Results saved to data/predictions/final_predictions.json")
