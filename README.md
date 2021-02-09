# Multilabel Classification with Huggingface’ 🤗 Trainer 💪 and AdapterHub 🤖: A short Tutorial for Multilabel Classification with Language Models
If you are a fan of the HuggingFace API 🤗, you may have noticed the new trainer 💪 class (introduced in version 2.9):

```python
trainer = Trainer(
    model,
    args,
    train_dataset=TRAIN_DATA,
    eval_dataset=TEST_DATA,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

The trainer class provides an easy way to do the training process for tuning your own language model (e.g., BERT - if you have never heard of language models like BERT before, you should stop here first and look at this amazing blog post) in a few lines of code, with all the options to customize the training (check out an example provided by HuggingFace for sentence classification).

The official tutorial only provides you a way to use it without the new trainer 💪 class. I will show in this notebook a small example using AdapterHub 🤖 doing the job for you by providing a multilabel head out-of-the box.

| type        | notebook           |
| ------------- |:-------------:|
| adapter      | xxx |
| multi-BERTmodel     |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aF4WeuNYDsIVWnp2xYMTgeRGQOAodO1g?usp=sharing)     |

You can also use the fast lane 🚀 by importing the `MultilabelTransformer` provided in this repository.
```python
from MultilabelTransformer import MultilabelRobertaForSequenceClassification

model = MultilabelRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=N)
```

_Note: `MultilabelTransformer` currently support `MultilabelRobertaForSequenceClassification` and `MultilabelBertForSequenceClassification`._

Happy Researching!
