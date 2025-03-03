import os
import json
from typing import Tuple, Dict
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

def load_model_and_tokenizer() -> Tuple[BertForSequenceClassification, BertTokenizerFast, Dict[str, int]]:
    """
    Load the model, tokenizer, and language dictionary from local files or pretrained model.
    """
    if os.path.exists("data/tokenizer"):
        return _load_from_local()
    return _load_from_pretrained()

def _load_from_local() -> Tuple[BertForSequenceClassification, BertTokenizerFast, Dict[str, int]]:
    with open("data/languages.json") as f:
        langs_dict = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained("data/tokenizer")

    with open("data/model_config.json") as f:
        config_data: dict = json.load(f)

    config_data.update({
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": tokenizer.vocab_size,
        "num_labels": len(langs_dict),
        "label2id": langs_dict,
        "id2label": {v: k for k, v in langs_dict.items()},
    })
    config = BertConfig(**config_data)

    return BertForSequenceClassification(config), tokenizer, langs_dict

def _load_from_pretrained() -> Tuple[BertForSequenceClassification, BertTokenizerFast, Dict[str, int]]:
    existing_model = "alexneakameni/language_detection"
    tokenizer = BertTokenizerFast.from_pretrained(existing_model)
    config = BertConfig.from_pretrained(existing_model)
    model = BertForSequenceClassification.from_pretrained(existing_model)
    return model, tokenizer, config.label2id
