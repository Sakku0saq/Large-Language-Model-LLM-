---
title: GPT From Scratch
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# Building a Large Language Model From Scratch

A GPT-style language model implemented from first principles in PyTorch —
tokenizer, self-attention, transformer blocks, pretraining, and fine-tuning —
with a live interactive demo.

## What's inside

- `gpt_model/model.py` — the GPT architecture (multi-head attention, layer
  norm, GELU feed-forward, transformer blocks), hand-written, no shortcuts.
- `gpt_model/generate.py` — greedy / temperature / top-k text generation.
- `gpt_model/hf_weights.py` — loads the official pretrained GPT-2 weights
  from the Hugging Face Hub into the custom architecture above.
- `gpt_model/spam_classifier.py` — fine-tunes a classification head on top
  of GPT-2 to detect spam (SMS Spam Collection dataset).
- `app.py` — the Streamlit app: a project write-up plus two live demos
  (text generation and spam classification).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first time you open the "Spam Classifier" tab, it trains the
classification head live (a couple of minutes on CPU) and caches the result
for the rest of the session.

## Deploying to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space, SDK = **Streamlit**.
2. Push these files to the Space's git repo (or upload them via the web UI).
3. The Space will build automatically using `requirements.txt` and launch
   `app.py`.
