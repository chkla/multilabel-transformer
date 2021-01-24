# Multilabel Classification with Huggingface’ 🤗 Trainer 💪 and AdapterHub 🤖: A short Tutorial for Multilabel Classification with Language Models
If you are a fan of the HuggingFace API 🤗, you may have noticed the new  [trainer 💪 class] (introduced in version 2.9):

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

The trainer class provides an easy way to do the training process for tuning your own language model (e.g., BERT - if you have never heard of language models like BERT before, you should stop here first and look at this amazing [blog post]) in a few lines of code, with all the options to customize the training (check out an example provided by [HuggingFace for sentence classification]).
