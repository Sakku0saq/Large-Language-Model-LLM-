# Attention Mechanisms from Scratch

This notebook implements the core attention mechanisms used in Transformer-based Large Language Models (LLMs) from first principles using PyTorch. Instead of relying on prebuilt attention modules, the notebook progressively builds attention from manual Query-Key-Value computations to complete reusable PyTorch modules.

The goal of this project is to understand how modern Transformer architectures process contextual information through attention.

---

## What This Notebook Covers

### 1. Self-Attention Fundamentals

The notebook begins with a manual implementation of self-attention by computing:

* Query vectors (Q)
* Key vectors (K)
* Value vectors (V)

using trainable weight matrices.

Starting from token embeddings, attention scores are computed through dot products between queries and keys, followed by normalization using Softmax to generate attention weights.

Key concepts explored:

* Query, Key, and Value projections
* Attention score computation
* Attention weight normalization
* Context vector generation
* Scaled dot-product attention

---

### 2. Self-Attention Implementations

Two versions of self-attention are implemented:

#### SelfAttention_v1

A low-level implementation using manually defined trainable weight parameters.

This version focuses on understanding the mathematics behind attention and how trainable projection matrices transform embeddings into queries, keys, and values.

#### SelfAttention_v2

A cleaner implementation using PyTorch's:

```python
nn.Linear
```

layers for Query, Key, and Value projections.

This version demonstrates how self-attention is typically implemented in modern deep learning codebases.

---

### 3. Causal (Masked) Self-Attention

The notebook extends standard self-attention by introducing causal masking.

A lower-triangular mask is applied so that each token can only attend to itself and previous tokens, preventing information leakage from future positions.

This mechanism is fundamental to autoregressive language models such as GPT.

Topics covered:

* Causal masking
* Lower-triangular attention masks
* Masked attention scores
* Softmax after masking
* Attention dropout

---

### 4. Multi-Head Attention

The notebook implements Multi-Head Attention in two different ways.

#### MultiHeadAttentionWrapper

Creates multiple independent attention heads using:

```python
nn.ModuleList
```

where each head performs causal attention separately before concatenation.

This implementation helps visualize how multiple attention heads operate in parallel.

#### MultiHeadAttention

A more efficient Transformer-style implementation that:

* Projects inputs into Queries, Keys, and Values
* Splits embeddings into multiple heads
* Computes attention in parallel
* Concatenates head outputs
* Applies a final output projection

This closely mirrors the architecture used in modern Transformer models.

---

## Learning Progression

The notebook follows the following sequence:

```text
Token Embeddings
       ↓
Manual Query, Key, Value Computation
       ↓
Attention Scores
       ↓
Attention Weights
       ↓
Context Vectors
       ↓
SelfAttention_v1
       ↓
SelfAttention_v2
       ↓
Causal Self-Attention
       ↓
MultiHeadAttentionWrapper
       ↓
Transformer-Style MultiHeadAttention
```

This progression is designed to build intuition before moving toward production-style implementations.

---

## Technologies Used

* Python
* PyTorch
* Jupyter Notebook

---

## Concepts Demonstrated

* Trainable attention weights
* Query-Key-Value transformations
* Scaled dot-product attention
* Softmax normalization
* Context vector computation
* Attention masking
* Causal self-attention
* Multi-head attention
* Dropout regularization
* PyTorch module design

---

## Why This Project?

Most tutorials introduce attention through high-level APIs. This notebook focuses on understanding the underlying mechanics by implementing each component step-by-step.

By the end of the notebook, the reader can understand how Transformer attention layers operate internally and how self-attention evolves into the multi-head attention mechanism used in modern LLMs.

---

## References

* Attention Is All You Need (Vaswani et al., 2017)
* PyTorch Documentation
* Transformer Architecture Fundamentals
