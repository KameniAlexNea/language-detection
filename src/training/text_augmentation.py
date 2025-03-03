import random
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TextAugmentation:
    """Text augmentation class for language detection tasks."""

    remove_digits_prob: float = 0.1
    shuffle_words_prob: float = 0.5
    remove_words_prob: float = 0.2
    include_digits_prob: float = 0.3
    punctuation_prob: float = 0.2
    case_transform_prob: float = 0.3
    add_whitespace_prob: float = 0.2
    char_substitute_prob: float = 0.15

    # Common character substitutions for typos
    CHAR_SUBS: Dict[str, List[str]] = {
        "a": ["@", "4"],
        "e": ["3"],
        "i": ["1", "!"],
        "o": ["0"],
        "s": ["$"],
        "t": ["7"],
    }

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

        if random.random() < self.case_transform_prob:
            text = self._transform_case(text)

        if random.random() < self.add_whitespace_prob:
            text = self._add_random_whitespace(text)

        if random.random() < self.char_substitute_prob:
            text = self._substitute_chars(text)

        return text

    def _transform_case(self, text: str) -> str:
        """Randomly transform case of words or characters."""
        if random.random() < 0.5:
            # Transform whole words
            words = text.split()
            words = [w.upper() if random.random() < 0.3 else w.lower() for w in words]
            return " ".join(words)
        else:
            # Transform individual characters
            return "".join(
                c.upper() if random.random() < 0.3 else c.lower() for c in text
            )

    def _add_random_whitespace(self, text: str) -> str:
        """Add random whitespace between characters."""
        chars = list(text)
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < 0.1:
                chars.insert(i, " ")
        return "".join(chars)

    def _substitute_chars(self, text: str) -> str:
        """Substitute characters with common alternatives."""
        chars = list(text)
        for i, char in enumerate(chars):
            lower_char = char.lower()
            if lower_char in self.CHAR_SUBS and random.random() < 0.2:
                chars[i] = random.choice(self.CHAR_SUBS[lower_char])
        return "".join(chars)

    def batch_call(self, texts: List[str]) -> List[str]:
        """Apply text augmentation to a batch of texts."""
        return [self(text) for text in texts]
