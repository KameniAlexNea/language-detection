import torch
import json
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

import os

save_folder = "data/predictions"
os.makedirs(save_folder, exist_ok=True)

parser = argparse.ArgumentParser(description="Run predictions on the test dataset.")
parser.add_argument(
    "--model_name",
    type=str,
    default="alexneakameni/language_detection",
    help="Path to the model checkpoint",
)
args = parser.parse_args()
model_name: str = args.model_name

# Load your dataset
# test = datasets.load_from_disk("data/test_dataset")
# label_tag = "lang"

test = datasets.load_dataset("papluca/language-identification", split="test")
label_tag = "labels"


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
        return_tensors="pt",
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

from tqdm import tqdm

results = {"predictions": [], "expected": []}
for pos, batch in tqdm(enumerate(test.batch(batch_size))):
    prediction_batch = predict_batch(batch)
    results["predictions"].append(prediction_batch["predicted_label"])
    results["expected"].append(batch[label_tag])
    if pos % 100 == 0:
        with open(f"{save_folder}/final_predictions_base.json", "w") as f:
            json.dump(results, f)


# Save the final predictions to a JSON file
with open(f"{save_folder}/final_predictions_base.json", "w") as f:
    json.dump(results, f)

print(
    f"Prediction complete. Results saved to {save_folder}/final_predictions_base.json"
)
