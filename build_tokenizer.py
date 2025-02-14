import os
from glob import glob
import logging
import random

random.seed(41)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.warning(os.getpid())
data_folder = "data/batch"
vocab_size = 50257

files = glob(f"{data_folder}/batch*.txt")
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
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

logging.warning("Tokenizer")

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.BertNormalizer(strip_accents=True)]
)

logging.warning("Add normalizer")

# Set the pre-tokenizer (here we use simple whitespace splitting).
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

logging.warning("BertPreTokenizer")

tokenizer.decoder = decoders.WordPiece()

logging.warning("BPEDecoder")

# Define special tokens and instantiate a trainer.
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
trainer = trainers.WordPieceTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    show_progress=True,
    min_frequency=2,  # Minimum frequency for a token to be included
    continuing_subword_prefix="##",  # Similar to BERT
)
logging.warning("BpeTrainer")

# Assume 'training_corpus' is an iterator over your text examples.
tokenizer.train(files, trainer=trainer)

logging.warning("Training")

tokenizer.save("data/custom_tokenizer.json")

logging.warning("save tokenizer")

special_tokens = [
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
]
logging.warning(special_tokens)

tokenizer.post_processor = processors.BertProcessing(
    sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
    cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
)

logging.warning("BertProcessing")

from transformers import BertTokenizerFast

# Wrap your trained tokenizer in a PreTrainedTokenizerFast
custom_tokenizer = BertTokenizerFast(
    tokenizer_object=tokenizer,
)
custom_tokenizer.model_max_length = 1024
custom_tokenizer.padding_side = "right"  # or 'left'
custom_tokenizer.truncation_side = "right"  # or 'left'

# Save the tokenizer files
custom_tokenizer.save_pretrained("data/tokenizer")

logging.warning("BertTokenizerFast save")
