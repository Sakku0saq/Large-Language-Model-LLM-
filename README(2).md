# Building a Large Language Model From Scratch

A hands-on learning and research project focused on understanding how
Large Language Models work by implementing and experimenting with the
core components of a GPT-style language model in PyTorch.

> **Learning in public:** This repository documents the process of
> moving from tokenization and embeddings to attention, GPT
> architecture, training, classification, and instruction fine-tuning.

## 📌 Project Overview

The main goal of this project is not simply to use an existing LLM, but
to understand the mechanics behind one by building the important pieces
step by step.

The notebook progresses through:

-   Text preprocessing and tokenization
-   Token IDs and vocabulary construction
-   Special tokens
-   Byte Pair Encoding (BPE)
-   Input-target pair generation
-   Token and positional embeddings
-   Self-attention
-   Causal attention
-   Attention dropout
-   Multi-head attention
-   GPT architecture
-   Layer normalization
-   GELU feed-forward networks
-   Shortcut/residual connections
-   Text generation
-   Cross-entropy loss and perplexity
-   Training and validation loss
-   Decoding strategies
-   Temperature scaling
-   Top-k sampling
-   Saving and loading model weights
-   Loading pretrained weights
-   Classification fine-tuning
-   Spam classification
-   Instruction fine-tuning
-   Instruction-response evaluation

## 🧠 What I Implemented

### 1. Tokenization

The project begins by converting raw text into tokens and then mapping
tokens to integer IDs.

It explores:

-   Regex-based preprocessing
-   Vocabulary creation
-   Encoding and decoding
-   `<|endoftext|>` and `<|unk|>` special tokens
-   GPT-style BPE tokenization using `tiktoken`

### 2. Dataset Preparation

The notebook creates input-target sequences for next-token prediction.

A sliding-window approach is used to generate training examples:

``` text
Input:  [token_1, token_2, token_3, token_4]
Target: [token_2, token_3, token_4, token_5]
```

A PyTorch `Dataset` and `DataLoader` are then used to prepare batches
for training.

### 3. Embeddings

The project implements:

-   Token embeddings
-   Positional embeddings
-   Combined input embeddings

This provides the model with both semantic token representations and
information about token position.

### 4. Attention

One of the main research areas in the notebook is attention.

The implementation progresses from a simplified attention mechanism to:

-   Trainable self-attention
-   Scaled attention
-   Causal masking
-   Attention dropout
-   Multi-head attention

This helps demonstrate how a Transformer can determine which previous
tokens are relevant when predicting the next token.

### 5. GPT Architecture

A GPT-style Transformer is assembled from its fundamental components,
including:

-   Token embeddings
-   Positional embeddings
-   Layer normalization
-   GELU activation
-   Feed-forward neural networks
-   Shortcut/residual connections
-   Transformer blocks
-   Output projection

The model can then generate text autoregressively.

### 6. Training and Evaluation

The notebook explores language-model training using:

-   Cross-entropy loss
-   Perplexity
-   Training loss
-   Validation loss
-   PyTorch optimizers
-   Training loops
-   Loss visualization

Model weights can also be saved and loaded for later experiments.

### 7. Decoding Strategies

Text generation is explored using different decoding techniques:

-   Temperature scaling
-   Top-k sampling
-   Combined temperature + top-k sampling

These experiments show how decoding parameters influence the randomness
and diversity of generated text.

## 🎯 Fine-Tuning Experiments

The project goes beyond basic text generation and explores downstream
tasks.

### Classification Fine-Tuning

A pretrained language model is adapted for classification by adding a
classification head and fine-tuning it on supervised data.

### Spam Classification

The model is also tested as a spam classifier, with examples such as:

``` text
"You are a winner you have been specially selected to receive $1000 cash..."
→ spam
```

and

``` text
"Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
→ not spam
```

### Instruction Fine-Tuning

The notebook then moves toward instruction-following.

The workflow includes:

1.  Preparing an instruction dataset
2.  Converting examples into an Alpaca-style format
3.  Creating train/validation/test splits
4.  Building training batches
5.  Creating target token IDs
6.  Masking target tokens
7.  Loading a pretrained LLM
8.  Fine-tuning on instruction data
9.  Generating model responses
10. Saving generated responses
11. Evaluating model responses

## 🔬 Evaluation and Research Observations

One of the most useful parts of this project is comparing generated
responses with expected responses rather than judging the model only by
whether its output looks fluent.

For example, the model can produce a reasonable alternative answer for
an instruction such as rewriting a sentence using a simile.

However, another evaluation example demonstrates an important failure
mode: the model can generate a fluent-sounding response while failing to
answer the actual question correctly.

This highlights an important lesson from the experiments:

> **Fluent generation is not the same as factual correctness or
> instruction-following.**

The project therefore treats model failures as part of the learning and
research process.

## 🛠️ Tech Stack

-   Python
-   PyTorch
-   `tiktoken`
-   NumPy
-   Matplotlib
-   tqdm
-   psutil
-   Ollama
-   Google Colab / Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Install dependencies

The notebook uses Python packages including:

``` bash
pip install torch tiktoken numpy matplotlib tqdm psutil
```

Ollama is used later in the notebook for external model-based
evaluation.

### 3. Open the notebook

Open:

``` text
Creating_Large_Language_Model_from_Scratch_Final_Version(2).ipynb
```

The notebook is designed to be worked through sequentially because later
experiments depend on variables, classes, model definitions, and
datasets created earlier.

## 📂 Repository Structure

A simple repository structure can look like:

``` text
.
├── Creating_Large_Language_Model_from_Scratch_Final_Version(2).ipynb
├── README.md
├── data/
│   └── the-verdict.txt
├── models/
│   └── saved model weights
└── outputs/
    ├── generated responses
    └── evaluation results
```

The exact files may vary depending on how the notebook is organized when
the repository is published.

## 📊 Current Project Status

This is an ongoing learning and research project.

### Completed / Explored

-   [x] Tokenization
-   [x] Token IDs
-   [x] Special tokens
-   [x] BPE tokenization
-   [x] Input-target datasets
-   [x] Token embeddings
-   [x] Positional embeddings
-   [x] Self-attention
-   [x] Causal attention
-   [x] Multi-head attention
-   [x] GPT architecture
-   [x] Text generation
-   [x] Loss and perplexity
-   [x] Training loop
-   [x] Decoding strategies
-   [x] Model saving/loading
-   [x] Classification fine-tuning
-   [x] Spam classification
-   [x] Instruction fine-tuning
-   [x] Response generation
-   [x] Response evaluation

### 🔭 Future Experiments

Potential next steps include:

-   Improving the training dataset and data pipeline
-   Running longer training experiments
-   Comparing different hyperparameters
-   Tracking training/validation metrics systematically
-   Improving instruction-following performance
-   Building a more robust evaluation pipeline
-   Investigating hallucination and factuality failures
-   Comparing decoding strategies quantitatively
-   Experimenting with larger models and datasets
-   Building a simple interface for interacting with the trained model

## 📚 Why This Project?

Modern LLMs can feel like black boxes when they are accessed only
through APIs.

Building the components from scratch provides a different perspective:

``` text
Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Embeddings
   ↓
Self-Attention
   ↓
Multi-Head Attention
   ↓
Transformer Blocks
   ↓
Next-Token Prediction
   ↓
Training
   ↓
Text Generation
   ↓
Fine-Tuning
   ↓
Evaluation
```

The purpose of this repository is to make that learning process visible
and reproducible.

## ⚠️ Important Note

This project is primarily educational and experimental. It is intended
to understand LLM architecture, training, fine-tuning, and evaluation
rather than to compete with production-scale language models.

The results shown in the notebook should therefore be interpreted as
experiments and learning milestones rather than benchmark claims.

## 🤝 Contributions

Suggestions, corrections, research ideas, and discussions are welcome.

If you find an implementation issue or have an idea for an experiment,
feel free to open an issue or submit a pull request.

## ⭐ Learning in Public

This repository is part of my journey to understand AI systems from the
inside out.

I am documenting not only successful experiments, but also incorrect
predictions, evaluation failures, implementation challenges, and lessons
learned along the way.

**Build → Experiment → Evaluate → Learn → Repeat.**

------------------------------------------------------------------------

### Author

**Saqlain**

Learning and researching Large Language Models from first principles.
