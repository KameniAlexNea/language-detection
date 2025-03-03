import torch

MODEL_CONFIG = {
    "model_name": "alexneakameni/language_detection",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

GRADIO_CONFIG = {
    "title": "🌍 Language Detection",
    "description": "Detects the language of a given text using a fine-tuned BERT model. Returns the top-k most probable languages."
}
