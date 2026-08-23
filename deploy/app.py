import time

import streamlit as st
import tiktoken
import torch

from gpt_model.hf_weights import load_pretrained_gpt2
from gpt_model.generate import generate, text_to_token_ids, token_ids_to_text
from gpt_model.spam_classifier import build_and_train_classifier, classify_text

st.set_page_config(page_title="GPT From Scratch", page_icon="🧠", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


@st.cache_resource(show_spinner=False)
def get_base_gpt2():
    model, cfg = load_pretrained_gpt2("gpt2")
    model.to(DEVICE)
    return model, cfg


@st.cache_resource(show_spinner=False)
def get_spam_classifier():
    tokenizer = get_tokenizer()
    base_model, cfg = load_pretrained_gpt2("gpt2")
    base_model.to(DEVICE)

    progress = st.progress(0.0, text="Training spam-classifier head on GPT-2 (one-time setup)...")

    def cb(step, total, loss):
        progress.progress(min(step / total, 1.0), text=f"Training... step {step}/{total}, loss {loss:.3f}")

    model, max_len, acc = build_and_train_classifier(
        base_model, cfg, tokenizer, DEVICE,
        num_epochs=3, batch_size=8, max_length=64,
        progress_callback=cb,
    )
    progress.empty()
    return model, max_len, acc


st.title("🧠 Building a Large Language Model From Scratch")
st.caption("Tokenizer → self-attention → GPT architecture → pretraining → fine-tuning, all implemented from first principles in PyTorch.")

tab_about, tab_generate, tab_classify = st.tabs(
    ["📖 About This Project", "✍️ Text Generation Demo", "📩 Spam Classifier Demo"]
)

# ---------------------------------------------------------------------------
# TAB 1: About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
## What this project demonstrates

This project is a **complete GPT-style large language model, implemented entirely from scratch** —
no `transformers` library shortcuts for the model itself, no black boxes. Every core LLM
component was hand-built and tested:

- **Custom tokenizer** — from a basic regex-based word tokenizer up to full byte-pair encoding (BPE)
- **Token & positional embeddings**
- **Self-attention from first principles** — starting with simplified, non-trainable attention
  and building up to scaled dot-product **causal multi-head attention** with dropout masking
- **Full transformer block** — layer normalization, GELU-activated feed-forward network, and
  residual/shortcut connections
- **The GPT model itself** — stacking transformer blocks into a configurable decoder-only
  architecture (this demo uses the 124M-parameter configuration, matching GPT-2 small)
- **Pretraining loop** — next-token prediction with cross-entropy loss, perplexity tracking,
  and train/validation loss curves
- **Decoding strategies** — greedy decoding, temperature scaling, and top-k sampling
- **Loading OpenAI's official pretrained GPT-2 weights** into the custom architecture, to
  verify it exactly reproduces GPT-2's behavior
- **Fine-tuning for classification** — freezing the base model and training a classification
  head on top of it to build a working **spam detector**
- **Instruction fine-tuning** — reformatting a dataset into Alpaca-style prompts and fine-tuning
  the model to follow instructions, with automated evaluation

### Why this matters
Most ML engineers use `AutoModel.from_pretrained(...)` and never look inside. This project
is the opposite: it opens up the transformer and rebuilds it piece by piece, which is why
this Space can load the *real* GPT-2 weights into *hand-written* attention and feed-forward
code and get identical outputs to the original model.

### Try it
- **Text Generation Demo** — the custom GPT model, loaded with real pretrained GPT-2 weights,
  generating text live from your prompt.
- **Spam Classifier Demo** — the same architecture, fine-tuned (right here, live, the first
  time this Space is loaded) into a working spam/ham classifier on the SMS Spam Collection dataset.
        """
    )

# ---------------------------------------------------------------------------
# TAB 2: Text generation
# ---------------------------------------------------------------------------
with tab_generate:
    st.subheader("Generate text with the from-scratch GPT model")
    st.write(
        "This loads OpenAI's real GPT-2 (124M) weights into the GPT architecture "
        "built from scratch in this project — same weights, hand-written model code."
    )

    prompt = st.text_input("Prompt", value="Every effort moves you")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_new_tokens = st.slider("Tokens to generate", 5, 100, 30)
    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1)
    with col3:
        top_k = st.slider("Top-k", 1, 100, 40)

    if st.button("Generate", type="primary"):
        with st.spinner("Loading GPT-2 weights (first run only)..."):
            model, cfg = get_base_gpt2()
            tokenizer = get_tokenizer()

        with st.spinner("Generating..."):
            torch.manual_seed(int(time.time()))
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(prompt, tokenizer).to(DEVICE),
                max_new_tokens=max_new_tokens,
                context_size=cfg["context_length"],
                temperature=temperature,
                top_k=top_k,
            )
            output_text = token_ids_to_text(token_ids, tokenizer)

        st.text_area("Output", output_text, height=150)

# ---------------------------------------------------------------------------
# TAB 3: Spam classifier
# ---------------------------------------------------------------------------
with tab_classify:
    st.subheader("Spam classifier, fine-tuned from the same GPT-2 base")
    st.write(
        "This takes the pretrained GPT-2 model, freezes almost all of it, adds a small "
        "classification head, and fine-tunes it on the SMS Spam Collection dataset — "
        "the exact recipe from the notebook's fine-tuning section. "
        "**First load trains the model live (a couple of minutes); after that it's instant.**"
    )

    message = st.text_area(
        "Message to classify",
        value="Congratulations! You've been selected to win a $1000 prize. Reply now to claim.",
        height=100,
    )

    if st.button("Classify", type="primary"):
        with st.spinner("Preparing classifier (training on first load)..."):
            model, max_len, acc = get_spam_classifier()
            tokenizer = get_tokenizer()

        label, spam_prob = classify_text(message, model, tokenizer, DEVICE, max_len)

        st.metric("Prediction", label.upper())
        st.progress(spam_prob, text=f"Spam probability: {spam_prob*100:.1f}%")
        st.caption(f"Classifier test-set accuracy this session: {acc*100:.1f}%")

st.divider()
st.caption(
    "Built end-to-end from scratch in PyTorch — tokenizer, attention, transformer blocks, "
    "pretraining, and fine-tuning — then deployed here as a live, interactive demo."
)
