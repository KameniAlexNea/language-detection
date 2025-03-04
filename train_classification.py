import logging
import os
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.training.text_augmentation import TextAugmentation
from src.training.data_loader import load_split_data
from src.training.model_config import load_model_and_tokenizer
from src.training.evaluator import ComputeMetric

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
TEST_BATCH_SIZE = 512
TRAIN_BATCH_SIZE = 256

# Environment setup
os.environ.update({
    "TOKENIZERS_PARALLELISM": "true",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "WANDB_PROJECT": "lang_detection"
})

def get_training_args() -> TrainingArguments:
    """Configure and return training arguments."""
    return TrainingArguments(
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
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TEST_BATCH_SIZE,
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
        metric_for_best_model="eval_loss",
    )

def main():
    # Load model and tokenizer
    model, tokenizer, langs_dict = load_model_and_tokenizer()
    
    # Load dataset
    train, valid, test = load_split_data(langs_dict, TEST_BATCH_SIZE)
    
    logging.info(f"Model configuration:\n{model.config}")
    
    # Setup augmentation and transformations
    augment_text = TextAugmentation()
    
    def transform(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokens["label"] = torch.tensor([langs_dict[i] for i in examples["label"]]
            if isinstance(examples["label"], list)
            else langs_dict[examples["label"]])
        return tokens

    def make_augmented_text(examples):
        examples["text"] = (augment_text.batch_call(examples["text"])
            if isinstance(examples["text"], list)
            else augment_text(examples["text"]))
        return transform(examples)

    # Apply transformations
    train.set_transform(make_augmented_text)
    valid.set_transform(transform)
    test.set_transform(transform)

    logging.info(f"Dataset Sizes - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train,
        eval_dataset=valid,
        processing_class=tokenizer,
        compute_metrics=ComputeMetric(langs_dict).compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train and evaluate
    trainer.evaluate(valid)
    trainer.train()
    trainer.evaluate(test, metric_key_prefix="test")

if __name__ == "__main__":
    main()
