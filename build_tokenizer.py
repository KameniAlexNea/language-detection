import os
from glob import glob
import logging
import random

random.seed(41)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.warning(os.getpid())

files = glob("data/dataset/batch/batch*.txt")
logging.warning(files)
logging.warning("Files Listed")

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    processors,
    normalizers,
)


logging.warning("Load end")
# Initialize an empty BPE tokenizer with an unknown token.
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

logging.warning("Tokenizer")

tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.BertNormalizer(strip_accents=True)])

logging.warning("Add normalizer")

# Set the pre-tokenizer (here we use simple whitespace splitting).
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

logging.warning("BertPreTokenizer")

tokenizer.decoder = decoders.BPEDecoder()

logging.warning("BPEDecoder")

# Define special tokens and instantiate a trainer.
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
trainer = trainers.BpeTrainer(
    vocab_size=50257,
    special_tokens=special_tokens,
    show_progress=True,
    min_frequency=2,  # Minimum frequency for a token to be included
    continuing_subword_prefix="##",  # Similar to BERT
)
logging.warning("BpeTrainer")

# Assume 'training_corpus' is an iterator over your text examples.
tokenizer.train(files, trainer=trainer)

logging.warning("Training")

tokenizer.save("custom_tokenizer.json")

logging.warning("save tokenizer")

special_tokens = [
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
]
logging.warning(special_tokens)

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=special_tokens,
)

logging.warning("TemplateProcessing")

from transformers import PreTrainedTokenizerFast

# Wrap your trained tokenizer in a PreTrainedTokenizerFast
custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    # model_max_length=512
)
custom_tokenizer.model_max_length = 512
custom_tokenizer.padding_side = "right"  # or 'left'
custom_tokenizer.truncation_side = "right"  # or 'left'

# Save the tokenizer files
custom_tokenizer.save_pretrained("data/tokenizer")

logging.warning("PreTrainedTokenizerFast save")
