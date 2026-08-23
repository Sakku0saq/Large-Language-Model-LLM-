# Building a Large Language Model From Scratch

🔗 **Live demo:** https://7prc7qurlmqwnhkmcmbbse.streamlit.app/

A GPT-style language model implemented from first principles in PyTorch — tokenizer, self-attention, transformer blocks, pretraining, and fine-tuning — with a live interactive demo.

## What's inside

- `gpt_model/model.py` — the GPT architecture (multi-head attention, layer norm, GELU feed-forward, transformer blocks), hand-written, no shortcuts.
- `gpt_model/generate.py` — greedy / temperature / top-k text generation.
- `gpt_model/hf_weights.py` — loads the official pretrained GPT-2 weights from the Hugging Face Hub into the custom architecture above.
- `gpt_model/spam_classifier.py` — fine-tunes a classification head on top of GPT-2 to detect spam (SMS Spam Collection dataset).
- `app.py` — the Streamlit app: a project write-up plus two live demos (text generation and spam classification).

## Try it live

Open the link above and check out:
- **Text Generation Demo** — the custom GPT model, loaded with real pretrained GPT-2 weights, generating text from any prompt.
- **Spam Classifier Demo** — the same architecture, fine-tuned into a working spam/ham classifier (trains live on first load, then cached).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first time you open the "Spam Classifier" tab, it trains the classification head live (a couple of minutes on CPU) and caches the result for the rest of the session.

## Deploying to Streamlit Community Cloud (free)

1. Push these files to a public GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Point it at your repo, branch, and `app.py`, then click **Deploy**.
4. You'll get a public URL like `your-app-name.streamlit.app`.

Note: the free tier has ~1GB RAM and apps sleep after inactivity (they wake back up automatically on the next visit).
