# NanoGPT from Scratch

A complete GPT-style Transformer language model implemented from scratch in PyTorch, trained on the Tiny Shakespeare dataset.

This project was built as a deep learning and systems-level learning exercise and serves as the **capstone implementation concluding the Stanford CS224N (Natural Language Processing with Deep Learning, Spring 2024) course**.

The implementation closely follows conceptual guidance from Andrej Karpathy’s “Let’s build GPT from scratch / NanoGPT” tutorial, while all code is written independently with a focus on architectural understanding.

--------------------------------------------------

PROJECT CONTEXT AND MOTIVATION

This project represents the practical culmination of completing the **Stanford CS224N (Spring 2024) lecture series**, covering:

- Word embeddings
- Sequence modeling
- Attention mechanisms
- Transformers
- Optimization and regularization
- Modern NLP architectures

After completing the full CS224N curriculum, this project was designed to consolidate theoretical understanding by implementing a GPT-style Transformer end-to-end from first principles.

--------------------------------------------------

PROJECT OVERVIEW

This repository contains an end-to-end implementation of an autoregressive Transformer decoder (GPT-style), built incrementally from the ground up.

The project begins with tokenization and simple language modeling concepts and evolves into a full GPT architecture incorporating:

- Multi-head causal self-attention
- Feedforward (MLP) layers
- Residual connections
- Layer Normalization
- Dropout regularization
- Stacked Transformer blocks

The final model contains approximately **10 million parameters** and supports **CUDA-based training and inference**.

--------------------------------------------------

ACKNOWLEDGEMENTS

This project was inspired by and learned from:

1. **Stanford CS224N — Natural Language Processing with Deep Learning (Spring 2024)**
2. **Andrej Karpathy — “Let’s build GPT from scratch / NanoGPT”**

Usage notes:
- CS224N provided the theoretical and mathematical foundation
- Andrej Karpathy’s work served as an educational implementation reference
- All code in this repository was written independently

Full credit to **Stanford University** and **Andrej Karpathy** for their exceptional educational contributions.

--------------------------------------------------

KEY CONCEPTS IMPLEMENTED

- Character-level tokenization
- Autoregressive language modeling
- Token embeddings
- Positional embeddings
- Causal (decoder-style) self-attention
- Multi-head attention with output projection
- Feedforward (MLP) networks with 4x expansion
- Residual connections
- Pre-Norm Layer Normalization (GPT-style)
- Dropout regularization
- Stacked Transformer blocks
- Training and validation loss estimation
- Autoregressive text generation
- CUDA (GPU) support

--------------------------------------------------

MODEL ARCHITECTURE

Transformer Block (Pre-Norm GPT):

x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))

- Attention enables token-to-token communication
- Feedforward networks perform per-token computation
- Residual connections stabilize gradient flow
- Layer Normalization stabilizes activation distributions

Multiple blocks are stacked to increase depth and representational capacity.

--------------------------------------------------

FINAL MODEL CONFIGURATION

Embedding dimension (n_embd): 384  
Number of Transformer layers (n_layer): 6  
Number of attention heads (n_head): 6  
Context length (block_size): 256  
Batch size: 64  
Dropout: 0.2  
Optimizer: AdamW  
Learning rate: 3e-4  
Total parameters: ~10M  

--------------------------------------------------

PARAMETER SCALE

The majority of parameters arise from:
- Attention projections (Query, Key, Value)
- Multi-head attention output projections
- Feedforward MLP layers
- Token and positional embeddings

This places the model beyond a toy example and into a realistic Transformer scale.

--------------------------------------------------

DATASET

- Tiny Shakespeare
- Character-level language modeling
- 90% training / 10% validation split

--------------------------------------------------

TRAINING OBJECTIVE

The model is trained using next-token prediction with cross-entropy loss.

Validation loss is estimated by averaging over multiple randomly sampled batches to reduce noise and variance.

--------------------------------------------------

TEXT GENERATION

- Fully autoregressive decoding
- Causal masking (no future token access)
- Probabilistic sampling via multinomial sampling

--------------------------------------------------

CUDA SUPPORT

The model automatically runs on GPU if available.
All tensors, batches, and model parameters are placed on the same device.

--------------------------------------------------

PROJECT GOALS

- Translate CS224N theory into real systems
- Understand Transformers by building them from scratch
- Learn why each architectural component exists
- Avoid black-box abstractions
- Develop strong intuition about GPT internals

--------------------------------------------------

WHAT THIS PROJECT IS

- A real GPT-style decoder
- Architecturally correct
- Scalable and stable
- Grounded in CS224N theory
- Educationally rigorous

--------------------------------------------------

WHAT THIS PROJECT IS NOT

- A production-scale LLM
- A fine-tuned large language model
- A project using high-level Transformer libraries

--------------------------------------------------

POSSIBLE EXTENSIONS

- Weight tying between embeddings and LM head
- Rotary or ALiBi positional embeddings
- Key–Value caching for fast inference
- Flash Attention
- Mixed precision training
- Scaling law experiments

--------------------------------------------------

STATUS

Project complete.

Tagged release:
v1.0-gpt-from-scratch

--------------------------------------------------

LICENSE

This project is intended for educational and research purposes.

--------------------------------------------------

AUTHOR

Yashraj Sharma  
Computer Science | Natural Language Processing | Transformers
