import logging
import os
import json
import random
import re
import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)
from evaluator import ComputeMetric

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["WANDB_PROJECT"] = "lang_detection"

# Training parameters
test_batch_size = 512
train_batch_size = 256


class TextAugmentation:
    def __init__(self, remove_digits=0.1, shuffle_words=0.5, remove_words=0.2, include_digits=0.3):
        self.prob_remove_digits = remove_digits
        self.prob_shuffle_words = shuffle_words
        self.prob_remove_words = remove_words
        self.prob_include_digits = include_digits

    def __call__(self, text: str):
        if random.random() < self.prob_include_digits:
            text += str(random.randint(0, 10000))
        if random.random() < self.prob_remove_digits:
            text = re.sub(r'\d', '', text)
        if random.random() < self.prob_shuffle_words:
            words = text.split()
            random.shuffle(words)
            text = " ".join(words)
        if random.random() < self.prob_remove_words and len(text.split()) > 1:
            words = text.split()
            words.pop(random.randint(0, len(words) - 1))
            text = " ".join(words)
        return text

    def batch_call(self, texts: list):
        return [self(text) for text in texts]


def load_split_data(langs_dict):
    ds = load_dataset("hac541309/open-lid-dataset", split="train")
    splits = ds.train_test_split(test_size=0.1, seed=41)
    train, test = splits["train"], splits["test"]
    test = test.train_test_split(test_size=len(langs_dict) * test_batch_size, seed=41)
    valid, test = test["test"], test["train"]

    test_save_path = "data/test_dataset"
    if not os.path.exists(test_save_path):
        logging.warning("Saving Test Set...")
        test.save_to_disk(test_save_path)
        logging.warning("Test Set Saved.")
    
    return train, valid, test


def load_model_and_tokenizer():
    model_path = "data/model"
    if os.path.exists(model_path):
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
            "id2label": {v: k for k, v in langs_dict.items()}
        })
        config = BertConfig(**config_data)

        model = BertForSequenceClassification(config)
        return model, tokenizer, langs_dict

    existing_model = "alexneakameni/language_detection"
    tokenizer = BertTokenizerFast.from_pretrained(existing_model)
    config = BertConfig.from_pretrained(existing_model)
    model = BertForSequenceClassification(config)
    return model, tokenizer, model.config.label2id


def main():
    # Load model and tokenizer
    model, tokenizer, langs_dict = load_model_and_tokenizer()

    # Load dataset
    train, valid, test = load_split_data(langs_dict)

    logging.info(model)

    augment_text = TextAugmentation()

    def make_augmented_text(examples):
        examples["text"] = augment_text.batch_call(examples["text"]) if isinstance(examples["text"], list) else augment_text(examples["text"])
        return transform(examples)

    def transform(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", return_tensors="pt")
        tokens["label"] = torch.tensor([langs_dict[i] for i in examples["label"]]) if isinstance(examples["label"], list) else torch.tensor(langs_dict[examples["label"]])
        return tokens

    # Apply transformations
    for dataset in [train, valid, test]:
        dataset = dataset.rename_column("lang", "label")

    train.set_transform(make_augmented_text)
    valid.set_transform(transform)
    test.set_transform(transform)

    logging.info(f"Dataset Sizes - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Define training arguments
    training_args = TrainingArguments(
        run_name="lang_detect",
        output_dir="./data/results",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=2e-5,
        auto_find_batch_size=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./data/logs",
        report_to="wandb",
        remove_unused_columns=False,
        data_seed=41,
        dataloader_num_workers=8,
        lr_scheduler_type="cosine",
        bf16=True,
        torch_compile=True,
        save_total_limit=10,
    )

    metric = ComputeMetric(langs_dict)

    # Initialize and train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=tokenizer,
        compute_metrics=metric.compute_metrics,
    )

    trainer.evaluate(valid)
    trainer.train()
    trainer.evaluate(test, metric_key_prefix="test")


if __name__ == "__main__":
    main()
