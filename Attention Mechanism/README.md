# Attention Mechanisms

This module implements the fundamental attention mechanisms used in Large Language Models (LLMs) and Transformer architectures. The notebook provides a hands-on, from-scratch implementation of self-attention variants using PyTorch, helping build intuition for how modern language models process and contextualize information.

## Overview

Attention is the core building block of Transformer-based models such as GPT, BERT, LLaMA, and other modern LLMs. It enables tokens in a sequence to selectively focus on relevant information from other tokens, allowing the model to capture contextual relationships effectively.

This notebook covers:

* Self-Attention with Trainable Weights
* Causal (Masked) Self-Attention
* Multi-Head Attention

## Contents

### Self-Attention with Trainable Weights

Implementation of the standard self-attention mechanism where learnable weight matrices generate:

* Queries (Q)
* Keys (K)
* Values (V)

The attention output is computed using scaled dot-product attention:

```math
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Key concepts explored:

* Query-Key similarity
* Attention score computation
* Softmax normalization
* Context vector generation
* Learnable projection matrices

---

### Causal Self-Attention

Causal attention ensures that each token can only attend to itself and preceding tokens.

This masking strategy is essential for autoregressive language models such as GPT because it prevents information leakage from future tokens during training.

Concepts covered:

* Attention masking
* Lower-triangular causal masks
* Masked attention scores
* Dropout integration

---

### Multi-Head Attention

Multi-head attention extends self-attention by allowing the model to learn multiple contextual representations simultaneously.

Benefits include:

* Capturing different semantic relationships
* Learning multiple attention patterns
* Improved representation learning
* Enhanced model capacity

Implementation includes:

* Multiple attention heads
* Parallel attention computation
* Head concatenation
* Output projection layers

---

## Learning Goals

After completing this notebook, you should understand:

* How self-attention works internally
* Why attention scores are scaled
* The purpose of causal masking
* How multi-head attention improves Transformer performance
* The attention mechanism used inside modern LLMs

## Technologies

* Python
* PyTorch
* Jupyter Notebook

## File

```text
SELF_ATTENTION_WITH_TRAINABLE_WEIGHTS_,_Casual_Self_Attention_,_Multi_Head_Attention.ipynb
```

## References

* Attention Is All You Need (Vaswani et al., 2017)
* PyTorch Documentation
* Transformer Architecture

## Position in the LLM Pipeline

```text
Input Tokens
      ↓
Token Embeddings
      ↓
Self-Attention
      ↓
Multi-Head Attention
      ↓
Feed Forward Network
      ↓
Transformer Blocks
      ↓
Large Language Model Output
```

This notebook focuses specifically on the attention layer, which serves as the foundation of modern Transformer-based language models.

