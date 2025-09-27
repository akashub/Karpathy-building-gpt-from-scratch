# **A Guided Journey to Building GPT From Scratch**

Welcome\! This project documents the step-by-step creation of a GPT-like Transformer model from scratch. The repository contains an evolutionary series of Python scripts and detailed markdown explanations.

To get the most out of this, follow the learning path below. Each chapter builds on the last, mirroring the process of building the model itself.

#### This entire project is heavily inspired by and follows the structure of Andrej Karpathy's wonderful [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY). A huge thank you to the OG for creating such an accessible and insightful resource.


## Chapter 1: The Foundation - A Simple Bigram Model

We begin with the simplest possible language model to understand the core task: predicting the next character in a sequence.

  * **Goal**: Learn how to build a model that predicts the next character based *only* on the current character, using a simple lookup table.
  * **Primary Script**: `bigram.py`
  * **Key Document**: `gpt-dev.ipynb` (the first half)

### 1.1 Data Preparation and Tokenization

A neural network understands numbers, not text. Our first job is to convert the *Tiny Shakespeare* dataset into a format the model can use.

  * **Tokenization**: We create a vocabulary of all unique characters in the text. Then, we create a mapping from each character to a unique integer (`stoi`) and back (`itos`). This process is called tokenization.

<!-- end list -->

```python
# Getting all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (and vice-versa)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

  * **Data Batching**: We don't train on the whole text at once. We feed the model random chunks (`batches`) of text. For each chunk of text (the input `x`), the model's goal is to predict the next character at every position (the target `y`). Therefore, `y` is just `x` shifted one position to the right.

<!-- end list -->

```python
# From get_batch function
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
```

### 1.2 The Bigram Model: A Giant Lookup Table

The `BigramLanguageModel` in `bigram.py` is the simplest possible approach.

  * **Concept**: It uses an `nn.Embedding` layer as a direct lookup table. This table has `vocab_size` rows and `vocab_size` columns.
      * When the model receives the index for a character (e.g., 'H'), it looks up the corresponding row.
      * That row contains `vocab_size` numbers, which are directly interpreted as the **logits** (raw scores) for every possible *next* character.

<!-- end list -->

```python
# From bigram.py
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table is both the lookup and the prediction layer
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # Shape: (B, T, vocab_size)
        # ... loss calculation ...
        return logits, loss
```

**Limitation**: This model has no real "understanding." The knowledge about what follows 'H' is completely independent of the knowledge about what follows 'T'. The model can't learn that vowels behave similarly or that some characters are just capitalized versions of others.

-----

## Chapter 2: The Leap to Neural Networks - Embeddings & Position

To overcome the Bigram model's limitations, we introduce an intermediate "thinking space" for the model.

  * **Goal**: Decouple a token's identity from the prediction of the next token. This allows the model to learn a rich, compressed representation of each character.
  * **Primary Script**: `v2.py`
  * **Key Documents**: `bigram_to_nn_tranformer.md`, `shapes.md`

### 2.1 The Two-Step Process

Instead of a direct lookup, we now use a two-step process:

1.  **Embedding**: Convert a token ID into a rich feature vector.
2.  **Prediction**: Use that feature vector to predict the next token.

### 2.2 Token and Positional Embeddings

  * **Token Embedding (`token_embedding_table`)**: This layer now maps each of the `vocab_size` characters to a dense vector of size `n_embd`. This vector is a learned representation of the character's features (e.g., "is it a vowel?", "is it punctuation?").

      * `nn.Embedding(vocab_size, n_embd)`

  * **Positional Embedding (`positional_embedding_table`)**: Self-attention is blind to order. We must explicitly tell the model *where* each token is in the sequence. This layer creates a unique learned vector for each position from 0 to `block_size - 1`.

      * `nn.Embedding(block_size, n_embd)`

These two embeddings are added together to produce a final vector for each token that encodes both **what it is** and **where it is**.

```python
# From v2.py
tok_embd = self.token_embedding_table(idx)      # (B, T, n_embd)
pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
x = tok_embd + pos_embd                         # (B, T, n_embd)
```

### 2.3 The Language Model Head (`lm_head`)

This `nn.Linear` layer takes the combined, position-aware embedding vector and projects it back into the vocabulary space to produce the final logits.

  * `nn.Linear(n_embd, vocab_size)`: This layer takes an input of size `n_embd` and produces an output of size `vocab_size`.

**Outcome**: You now understand the fundamental structure of a modern language model: **Input -\> Embedding -\> Processing -\> Prediction**.

-----

## Chapter 3: The Breakthrough - How Tokens Communicate with Self-Attention

This is the core concept of the Transformer. We give the tokens a mechanism to exchange information and build context.

  * **Goal**: Understand the mechanics of self-attention, where each token can look at previous tokens to create a context-aware representation.
  * **Primary Script**: The `Head` class in `v2.py` and `v2_deepernn.py`.
  * **Key Documents**: `gpt-dev.ipynb`, `understanding_self_attention_code_block.md`, `qkv_tensors_vs_weights.md`

### 3.1 The Intuition: A Weighted Average

At its heart, attention is a sophisticated way to create a weighted average of past information. The `gpt-dev.ipynb` notebook builds this intuition perfectly:

1.  **The Goal (For-loops)**: The simplest way to give a token context is to average its vector with all the vectors that came before it.
2.  **The Trick (Matrix Multiplication)**: A much faster way to compute this rolling average is with a single matrix multiplication using a lower-triangular matrix.
3.  **The Real Deal (Softmax)**: Instead of a simple average, we use `softmax` to create a **learnable, weighted average**. The model can now *decide* which past tokens are most important.

### 3.2 The Mechanism: Query, Key, and Value

Each token's position-aware embedding (`x`) is projected into three new vectors:

  * **Query (Q)**: "What am I looking for?"
  * **Key (K)**: "What information do I contain?"
  * **Value (V)**: "What information will I provide if you attend to me?"

The attention process works as follows:

1.  **Compute Scores**: The **Query** of the current token is multiplied with the **Key** of every previous token to get an affinity score (`wei = q @ k.transpose(...)`).
2.  **Mask & Scale**: We mask out future tokens (using `masked_fill`) and scale the scores to stabilize training.
3.  **Normalize**: `softmax` converts these scores into attention weights (probabilities that sum to 1).
4.  **Aggregate**: The final representation is a weighted sum of the **Value** vectors of all tokens, using the attention weights.

<!-- end list -->

```python
# From the Head class in v2_deepernn.py
k = self.key(x)
q = self.query(x)
wei = q @ k.transpose(-2, -1) * C**-0.5
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = self.value(x)
out = wei @ v
```

**Outcome**: You can now explain how a sequence of isolated vectors is transformed into a sequence of context-aware vectors.

-----

## Chapter 4: The Final Blueprint - Assembling a Deep Transformer

Now we assemble all the components into the final, multi-layered model capable of learning complex patterns.

  * **Goal**: Stack multiple attention and computation layers to create a deep network.
  * **Primary Script**: The full `v2_deepernn.py`.
  * **Key Document**: `everything.md`.

### 4.1 Multi-Head Attention

Instead of just one attention mechanism, we use multiple "heads" in parallel. Each head can specialize in learning different types of relationships (e.g., one might track subjects and verbs, another might track nearby characters). The outputs of all heads are concatenated and projected back to the original embedding size.

### 4.2 The Transformer Block

The `Block` is the repeating unit of our model. It contains two main sub-layers:

1.  **A Multi-Head Self-Attention Layer**: This is the "communication" part.
2.  **A Feed-Forward Network**: This is the "computation" or "thinking" part, where the model processes the information gathered by attention.

Crucially, each of these sub-layers is wrapped with **Layer Normalization** and **Residual Connections (`x = x + ...`)**. These are essential for enabling the training of deep networks by preventing gradients from vanishing and stabilizing the learning process.

### 4.3 Stacking Blocks

The final model in `v2_deepernn.py` creates and stacks `n_layer` (e.g., 4) of these blocks. Stacking allows the model to build up progressively more complex and abstract representations of the text.

**Outcome**: You can now trace the data flow through the entire final model and understand the purpose of every component.

-----

## Chapter 5: The Data's Journey & Model Execution

This final chapter provides a high-level summary of the entire process and explains the modern optimizations used in the final script.

  * **Goal**: Consolidate your understanding by following a batch of data from start to finish and learn about the performance-enhancing features.
  * **Primary Script**: The full `v2_deepernn.py`.
  * **Key Document**: `journey_of_data.md` and `everything.md`.

### 5.1 End-to-End Data Flow

The `journey_of_data.md` document provides a definitive trace of the tensor shapes as they are transformed by each layer of the network, from the initial `(16, 128)` integer tensor to the final `(16, 128, 65)` logit tensor. It's the ultimate summary of the architecture in action.

### 5.2 Performance Optimizations

The final script includes two powerful, one-line additions:

  * **`torch.compile(model)`**: A just-in-time (JIT) compiler that analyzes the model's Python code and fuses operations into highly optimized GPU kernels, leading to significant speedups.
  * **`torch.utils.checkpoint`**: A memory-saving technique that trades a little extra computation for a massive reduction in GPU memory usage. It avoids storing all intermediate activations and recomputes them during the backward pass, enabling the training of much larger models.

**Outcome**: You are now familiar with the entire process, from data prep to the advanced techniques used to train modern Transformer models efficiently.