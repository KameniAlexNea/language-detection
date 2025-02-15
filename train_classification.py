import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import os
import json
import torch
import datasets
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)
from evaluator import compute_metrics

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["WANDB_PROJECT"] = "lang_detection"
# os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_WATCH"] = "none"


test_batch_size = 512
train_batch_size = 256

def load_split_data(langs_dict):
    ds = datasets.load_dataset("hac541309/open-lid-dataset", split="train")
    splits = ds.train_test_split(test_size=0.1, seed=41)
    train, test = splits["train"], splits["test"]
    test = test.train_test_split(test_size=len(langs_dict) * test_batch_size, seed=41)
    valid, test = test["test"], test["train"]

    logging.warning("Before Saving Test Set")
    folder_name = "data/test_dataset"
    if not os.path.exists(folder_name):
        test.save_to_disk(folder_name)
    logging.warning("Test Set Saved")
    return train, valid, test


def main():
    # Load language dictionary
    langs_dict: dict = json.load(open("data/languages.json"))

    # Load dataset
    train, valid, test = load_split_data(langs_dict)

    # Load tokenizer
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        "data/tokenizer"
    )

    # Load and update model configuration
    config_file = json.load(open("data/model_config.json"))
    config_file["pad_token_id"] = tokenizer.pad_token_id
    config_file["vocab_size"] = tokenizer.vocab_size
    config_file["num_labels"] = len(langs_dict)
    config_file["label2id"] = langs_dict
    config_file["id2label"] = {v: k for k, v in langs_dict.items()}
    config = BertConfig(**config_file)

    # Initialize model
    model = BertForSequenceClassification(config)

    logging.info(model)

    # Define transformation function
    def transform(examples):
        tokens = tokenizer(
            examples["text"], truncation=True, padding="max_length", return_tensors="pt"
        )
        tokens["label"] = (
            torch.tensor([langs_dict[i] for i in examples["label"]])
            if isinstance(examples["label"], list)
            else torch.tensor(langs_dict[examples["label"]])
        )
        return tokens

    # Rename columns and set transformation
    train = train.rename_column("lang", "label")
    test = test.rename_column("lang", "label")
    valid = valid.rename_column("lang", "label")
    train.set_transform(transform)
    valid.set_transform(transform)
    test.set_transform(transform)

    print(valid, test, train)

    print("Training size: ", len(train), len(valid), len(test))

    # Set training arguments
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
        # dataloader_drop_last=True,
        bf16=True,
        torch_compile=True,
        save_total_limit=10
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,  # Add compute_metrics to Trainer
    )

    trainer.evaluate(valid)

    # Start training
    trainer.train()

    trainer.evaluate(test, metric_key_prefix="test")


if __name__ == "__main__":
    main()
