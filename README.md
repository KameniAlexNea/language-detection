---
library_name: transformers
tags:
- language
- detection
- classification
license: mit
datasets:
- hac541309/open-lid-dataset
pipeline_tag: text-classification
---

# Language Detection Model

A **BERT-based** language detection model trained on [hac541309/open-lid-dataset](https://huggingface.co/datasets/hac541309/open-lid-dataset), which includes **121 million sentences across 200 languages**. This model is optimized for **fast and accurate** language identification in text classification tasks.

## Model Details

- **Architecture**: [BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html)
- **Hidden Size**: 384  
- **Number of Layers**: 4  
- **Attention Heads**: 6  
- **Max Sequence Length**: 512  
- **Dropout**: 0.1  
- **Vocabulary Size**: 50,257  

## Training Process

- **Dataset**: 
  - Used the [open-lid-dataset](https://huggingface.co/datasets/hac541309/open-lid-dataset)  
  - Split into train (90%) and test (10%)
- **Tokenizer**: A custom `BertTokenizerFast` with special tokens for `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`
- **Hyperparameters**:  
  - Learning Rate: 2e-5  
  - Batch Size: 256 (training) / 512 (testing)  
  - Epochs: 1  
  - Scheduler: Cosine  
- **Trainer**: Leveraged the Hugging Face [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) with Weights & Biases for logging

## Evaluation

The model was evaluated on the test split. Below are the overall metrics:

- **Accuracy**: 0.969466  
- **Precision**: 0.969586  
- **Recall**: 0.969466  
- **F1 Score**: 0.969417

Detailled evaluation (Size is the number of languages supported)

| Script | Support | Precision | Recall | F1 Score | Size |
|--------|---------|-----------|--------|----------|------|
| Arab   | 819219  | 0.9038    | 0.9014 | 0.9023   | 21   |
| Latn   | 7924704 | 0.9678    | 0.9663 | 0.9670   | 125  |
| Ethi   | 144403  | 0.9967    | 0.9964 | 0.9966   | 2    |
| Beng   | 163983  | 0.9949    | 0.9935 | 0.9942   | 3    |
| Deva   | 423895  | 0.9495    | 0.9326 | 0.9405   | 10   |
| Cyrl   | 831949  | 0.9899    | 0.9883 | 0.9891   | 12   |
| Tibt   | 35683   | 0.9925    | 0.9930 | 0.9927   | 2    |
| Grek   | 131155  | 0.9984    | 0.9990 | 0.9987   | 1    |
| Gujr   | 86912   | 0.99999   | 0.9999 | 0.99995  | 1    |
| Hebr   | 100530  | 0.9966    | 0.9995 | 0.9981   | 2    |
| Armn   | 67203   | 0.9999    | 0.9998 | 0.9998   | 1    |
| Jpan   | 88004   | 0.9983    | 0.9987 | 0.9985   | 1    |
| Knda   | 67170   | 0.9999    | 0.9998 | 0.9999   | 1    |
| Geor   | 70769   | 0.99997   | 0.9998 | 0.9999   | 1    |
| Khmr   | 39708   | 1.0000    | 0.9997 | 0.9999   | 1    |
| Hang   | 108509  | 0.9997    | 0.9999 | 0.9998   | 1    |
| Laoo   | 29389   | 0.9999    | 0.9999 | 0.9999   | 1    |
| Mlym   | 68418   | 0.99996   | 0.9999 | 0.9999   | 1    |
| Mymr   | 100857  | 0.9999    | 0.9992 | 0.9995   | 2    |
| Orya   | 44976   | 0.9995    | 0.9998 | 0.9996   | 1    |
| Guru   | 67106   | 0.99999   | 0.9999 | 0.9999   | 1    |
| Olck   | 22279   | 1.0000    | 0.9991 | 0.9995   | 1    |
| Sinh   | 67492   | 1.0000    | 0.9998 | 0.9999   | 1    |
| Taml   | 76373   | 0.99997   | 0.9999 | 0.9999   | 1    |
| Tfng   | 41325   | 0.8512    | 0.8246 | 0.8247   | 2    |
| Telu   | 62387   | 0.99997   | 0.9999 | 0.9999   | 1    |
| Thai   | 83820   | 0.99995   | 0.9998 | 0.9999   | 1    |
| Hant   | 152723  | 0.9945    | 0.9954 | 0.9949   | 2    |
| Hans   | 92689   | 0.9893    | 0.9870 | 0.9882   | 1    |


A detailed per-script classification report is also provided in the repository for further analysis.

---

### How to Use

You can quickly load and run inference with this model using the [Transformers pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("alexneakameni/language_detection")
model = AutoModelForSequenceClassification.from_pretrained("alexneakameni/language_detection")

language_detection = pipeline("text-classification", model=model, tokenizer=tokenizer)

text = "Hello world!"
predictions = language_detection(text)
print(predictions)
```

This will output the predicted language code or label with the corresponding confidence score.

---

**Note**: The model’s performance may vary depending on text length, language variety, and domain-specific vocabulary. Always validate results against your own datasets for critical applications. 

For more information, see the [repository documentation](https://github.com/KameniAlexNea/learning_language). 

Thank you for using this model—feedback and contributions are welcome!
