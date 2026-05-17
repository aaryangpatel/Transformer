# Transformer

A compact **encoder–decoder Transformer** in PyTorch for **English → Spanish** translation on the [OPUS Books](https://huggingface.co/datasets/opus_books) dataset.

## What’s included

- Model: embeddings, sinusoidal positional encoding, multi-head attention, FFN blocks, residual + layer norm
- Training: WordLevel tokenizers, CrossEntropy with label smoothing, TensorBoard logging
- Validation: greedy decode, BLEU / WER / CER (sample batches)

## Run

Install deps (`torch`, `datasets`, `tokenizers`, `tqdm`, `torchmetrics`, `tensorboard`), then:

```bash
python train.py
```

Checkpoints go under `opus_books_weights/`. Adjust batch size, epochs, and paths in `config.py`.

## Files

| File        | Role                          |
|------------|--------------------------------|
| `model.py` | Transformer blocks & `build_transformer` |
| `train.py` | Data, train loop, validation  |
| `dataset.py` | Bilingual batches & masks   |
| `config.py` | Hyperparameters             |
