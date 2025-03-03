import random
import re
from dataclasses import dataclass
from typing import List

@dataclass
class TextAugmentation:
    """Text augmentation class for language detection tasks."""
    remove_digits_prob: float = 0.1
    shuffle_words_prob: float = 0.5
    remove_words_prob: float = 0.2
    include_digits_prob: float = 0.3
    punctuation_prob: float = 0.2

    def __call__(self, text: str) -> str:
        """Apply text augmentation to a single text."""
        if random.random() < self.include_digits_prob:
            text += str(random.randint(0, 10000))
        if random.random() < self.remove_digits_prob:
            text = re.sub(r"\d", "", text)
        if random.random() < self.shuffle_words_prob:
            words = text.split()
            random.shuffle(words)
            text = " ".join(words)
        if random.random() < self.remove_words_prob and len(text.split()) > 1:
            words = text.split()
            words.pop(random.randint(0, len(words) - 1))
            text = " ".join(words)
        if random.random() < self.punctuation_prob:
            text = re.sub(r"[^\w\s]", "", text)
        return text

    def batch_call(self, texts: List[str]) -> List[str]:
        """Apply text augmentation to a batch of texts."""
        return [self(text) for text in texts]
