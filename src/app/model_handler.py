from transformers import pipeline
import pycountry
from config import MODEL_CONFIG

class LanguageDetector:
    def __init__(self):
        self.model = pipeline(
            "text-classification",
            model=MODEL_CONFIG["model_name"],
            device=0 if MODEL_CONFIG["device"] == "cuda" else -1
        )

    @staticmethod
    def get_language_name(code: str):
        lang = code.split("_")[0]
        try:
            return pycountry.languages.get(alpha_3=lang).name
        except AttributeError:
            return lang

    def predict_language(self, text: str, top_k: int = 5) -> str:
        results = self.model(text, top_k=top_k)
        formatted_results = [
            f"{self.get_language_name(result['label'])} - {result['label']}: {result['score']:.4f}"
            for result in results
        ]
        return "\n".join(formatted_results)
