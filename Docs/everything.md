
# Building a GPT from Scratch: A Deep Dive

This document provides a comprehensive walkthrough of a character-level GPT (Generative Pre-trained Transformer) model, inspired by Andrej Karpathy's "Let's build GPT" series. We will dissect the script, explaining each component's purpose (the "why") and its underlying mechanics (the "how"), with a focus on data shapes and linear algebra transformations.

## 📜 Table of Contents

1.  **Project Setup & Data Preparation**
    * Hyperparameters and Setup
    * Data Loading and Tokenization
    * Data Batching for Training

2.  **The Building Blocks of a Transformer**
    * The Self-Attention Head
    * Multi-Head Attention
    * Feed-Forward Network

3.  **Assembling the Transformer**
    * The `Block` Class (with LayerNorm and Residual Connections)

4.  **The Complete Language Model**
    * The `BigramLanguageModel` Class
    * The `forward` Pass: A Data Journey

5.  **Training, Evaluating, and Generating Text** (Your new combined section)
    * The Training Loop
    * Evaluating Model Performance (`estimate_loss`)
    * Generating New Text

6.  **Model Optimization and Execution**
    * `torch.compile(model)`
    * `torch.set_float32_matmul_precision('high')`

7.  **Conclusion**

---

## 1. Project Setup & Data Preparation

Before we can build our model, we need to set up our environment and prepare the data.

### Hyperparameters and Setup

These are the knobs and dials that control our model's architecture and the training process.

```python
# hyperparameters
batch_size = 16 # How many independent sequences will we process in parallel?
block_size = 128 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256 # Number of embedding dimensions for each token.
dropout = 0.2
n_head = 8 # Number of self-attention heads.
n_layer = 4 # Number of transformer blocks.
torch.set_float32_matmul_precision('high') # For modern GPU performance.
````

  * **`batch_size`**: The number of text chunks processed simultaneously. A larger batch size can stabilize training but requires more memory.
  * **`block_size`**: The context window. The model can see up to `128` characters into the past to predict the next one.
  * **`n_embd`**: The embedding dimension. Each token will be represented by a vector of `256` numbers. This is the model's primary "working space."
  * **`n_head`** & **`n_layer`**: These define the model's depth and the complexity of its attention mechanism. We'll stack `4` transformer blocks (`n_layer`), and each block's attention mechanism will have `8` parallel heads (`n_head`).

### Data Loading and Tokenization

A neural network doesn't understand characters like 'H' or 'e'. It only understands numbers. **Tokenization** is the process of converting a sequence of characters into a sequence of numbers.

```python
with open('tinyShakeSpeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (and vice-versa)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: list of integers -> string

# Convert the entire dataset into a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
```

Here, our "vocabulary" is simply the set of all unique characters in the text. `vocab_size` tells us how many unique characters there are. We create two dictionaries: `stoi` (string-to-integer) and `itos` (integer-to-string) to handle the conversion.

### Data Batching for Training

We don't train the model on the entire text at once. Instead, we feed it small, random chunks called **batches**. The `get_batch` function is responsible for this.

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for our batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create the input sequences (x)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create the target sequences (y), which are offset by one character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

  * **Concept**: For every input sequence of characters, the model must predict the *next* character at each position. Therefore, the target (`y`) is simply the input (`x`) shifted one position to the right.
  * **Computation**:
      * `ix` is a tensor of `batch_size` random integers. These are the starting indices for our text chunks.
      * `x` becomes a tensor of shape `(batch_size, block_size)`, e.g., `(16, 128)`.
      * `y` also becomes a tensor of shape `(batch_size, block_size)`.

-----

## 2\. The Building Blocks of a Transformer

Now we get to the core components that make the Transformer architecture so powerful.

### The Self-Attention Head (`Head` class)

**Concept**: Self-attention is the mechanism that allows tokens to "look" at other tokens in the sequence and decide which ones are most important for understanding the context. A single `Head` is one perspective on these relationships.

Imagine you're reading the sentence: "The cat sat on the mat." To understand what "it" refers to in a follow-up sentence like "It was happy," you'd pay attention to "cat." The attention head learns to do this automatically.

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
```

**Computation & Data Flow**:

1.  **Input**: The `forward` method receives a tensor `x` of shape `(B, T, C)`, where `B`=batch\_size, `T`=block\_size, and `C`=n\_embd. For example: `(16, 128, 256)`.

2.  **Create Q, K, V**: Each token needs to play three roles:

      * **Query (q)**: "I am token *A*. What should I pay attention to?"
      * **Key (k)**: "I am token *B*. Here's what I contain."
      * **Value (v)**: "I am token *B*. If you pay attention to me, this is the information I'll provide."

    We generate these using linear layers. If `head_size` is `32` (since `n_embd`/`n_head` = `256/8`), the shapes become:

      * `q = self.query(x)` -\> `(16, 128, 32)`
      * `k = self.key(x)`   -\> `(16, 128, 32)`
      * `v = self.value(x)` -\> `(16, 128, 32)`

3.  **Calculate Attention Scores**: We determine the affinity between each token's **Query** and every other token's **Key** using a scaled dot-product.

    ```python
    # (B, T, C) @ (B, C, T) -> (B, T, T)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    ```

      * `k.transpose(-2, -1)` swaps the last two dimensions of the key tensor, resulting in a shape of `(16, 32, 128)`.
      * The matrix multiplication `q @ k.transpose(...)` computes the dot product between every query and every key. The result, `wei`, is an affinity matrix of shape **`(16, 128, 128)`**. Each element `wei[b, i, j]` holds a score representing how much token `i` should attend to token `j` in batch `b`.
      * **Scaling**: We divide by the square root of the head size (`C**-0.5`). This prevents the dot products from becoming too large, which would result in tiny gradients after the softmax, making training unstable.

4.  **Masking**: Since this is a text *generator*, we can't let a token see into the future. The model must predict the next token using only the preceding ones. We enforce this with a causal mask.

    ```python
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    ```

      * `self.tril` is a lower-triangular matrix of ones.
      * `self.tril == 0` creates a boolean mask where all upper-triangular positions are `True`.
      * `masked_fill` replaces these `True` positions with negative infinity. When we apply softmax next, these positions will become zero.

5.  **Softmax**: We convert the attention scores into probabilities that sum to 1.

    ```python
    wei = F.softmax(wei, dim=-1)
    ```

6.  **Aggregate Values**: Finally, we compute a weighted sum of the **Value** vectors using our attention weights.

    ```python
    # (B, T, T) @ (B, T, C) -> (B, T, C)
    out = wei @ v
    ```

    The output `out` has a shape of `(16, 128, 32)`. Each token's new representation is a blend of information from all previous tokens in the sequence, weighted by their importance.

### Multi-Head Attention (`MultiHeadAttention` class)

**Concept**: One attention head might learn to focus on the preceding noun, while another might focus on verbs far away in the sequence. **Multi-Head Attention** runs multiple heads in parallel and concatenates their results, allowing the model to capture a richer variety of contextual relationships.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project the concatenated output back to the original embedding size
        out = self.dropout(self.proj(out))
        return out
```

**Computation & Data Flow**:

1.  **Parallel Heads**: `h(x)` is called for each of the `8` heads, each producing an output of shape `(16, 128, 32)`.
2.  **Concatenation**: `torch.cat` stitches these `8` outputs together along the last dimension (`dim=-1`).
      * The resulting shape is `(16, 128, 32 * 8)`, which is `(16, 128, 256)`.
3.  **Projection**: A final linear layer `self.proj` takes this concatenated tensor and projects it back to the original embedding dimension `(16, 128, 256)`. This projection layer allows the model to learn how to best combine the information from the different heads.

### Feed-Forward Network (`Feedforward` class)

**Concept**: After the tokens have gathered context via self-attention (communication), the Feed-Forward Network (FFN) provides a "computation" step. It processes the information gathered by each token individually, transforming it and preparing it for the next block.

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
```

**Computation**: The FFN is applied to each token's vector independently.

1.  **Input**: A tensor of shape `(B, T, n_embd)`, e.g., `(16, 128, 256)`.
2.  **Expansion**: The first linear layer expands the dimension from `n_embd` to `4 * n_embd` (`256 -> 1024`).
3.  **Activation**: A ReLU non-linearity is applied.
4.  **Contraction**: The second linear layer projects it back down to `n_embd` (`1024 -> 256`).
5.  **Output**: A tensor of the original shape `(16, 128, 256)`.

-----

## 3\. Assembling the Transformer (`Block` class)

A **Transformer Block** is the fundamental repeating unit of the model. It standardizes the process of communication and computation.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

**Architecture & Data Flow**:

1.  **Layer Normalization (`ln1`)**: Before attention, we normalize the input `x`. LayerNorm stabilizes training by ensuring the inputs to each layer have a consistent mean and variance.
2.  **Multi-Head Attention (`sa`)**: The normalized tensor goes through the multi-head attention mechanism ("communication").
3.  **Residual Connection (`x + ...`)**: The output of the attention mechanism is **added** to the original input `x`. This "skip connection" is crucial for training deep networks, as it allows gradients to flow directly through the network, preventing them from vanishing.
4.  **Layer Normalization (`ln2`)**: We normalize the result of the first residual connection.
5.  **Feed-Forward Network (`ffwd`)**: The normalized tensor goes through the FFN ("computation").
6.  **Second Residual Connection**: The output of the FFN is added to its input.

Crucially, a tensor of shape `(B, T, C)` that enters the block will have the exact same shape when it exits. This allows us to stack these blocks seamlessly.

-----

## 4\. The Complete Language Model

Now, we assemble all the components into the final `BigramLanguageModel`.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

### The `forward` Pass: A Data Journey

Let's trace the full journey of a batch of data.

1.  **Input `idx`**: A tensor of token integers.

      * Shape: `(B, T)` -\> `(16, 128)`

2.  **Embeddings**: We get embeddings for both the tokens and their positions.

      * `tok_embd = self.token_embedding_table(idx)` -\> Shape: `(16, 128, 256)`
      * `pos_embd = self.positional_embedding_table(torch.arange(T))` -\> Shape: `(128, 256)`
      * **Why both?** The token embedding tells us *what* each token is, and the positional embedding tells us *where* it is in the sequence. Self-attention itself has no inherent sense of order, so we must inject this information.

3.  **Combine Embeddings**:

      * `x = tok_embd + pos_embd` -\> Shape: `(16, 128, 256)`. PyTorch broadcasting automatically adds the positional embedding to every sequence in the batch.

4.  **Transformer Blocks**: The combined embeddings are processed through the stack of `n_layer` blocks.

      * `x = self.blocks(x)` -\> Shape remains `(16, 128, 256)`.

5.  **Gradient Checkpointing**:

    ```python
    x = cp.checkpoint_sequential(self.blocks, segments=2, input=x, use_reentrant=False)
    ```

      * **Concept**: This is a powerful memory-saving technique. Instead of storing all the intermediate values (activations) inside the `self.blocks` for the backward pass, it saves only a few ("checkpoints"). During backpropagation, it recomputes the other activations on the fly. This trades a bit of extra compute time for a significant reduction in GPU memory usage, allowing us to train larger models.

6.  **Final Normalization**:

      * `x = self.ln_f(x)` -\> Shape remains `(16, 128, 256)`.

7.  **Language Model Head**: The final linear layer projects the model's internal representation into a score for each possible next character in our vocabulary.

      * `logits = self.lm_head(x)` -\> Shape: `(B, T, vocab_size)` -\> `(16, 128, 65)`.

8.  **Loss Calculation**:

      * `logits` are reshaped to `(B*T, vocab_size)` -\> `(2048, 65)`.
      * `targets` are reshaped to `(B*T)` -\> `(2048)`.
      * `loss = F.cross_entropy(logits, targets)` calculates how well the model's predictions align with the actual next characters, giving us a single number to optimize.

-----

Of course\! Here are sections 5 and 7 combined into a single, streamlined part.

-----

## 5\. Training, Evaluating, and Generating Text 🚀

With the model fully assembled, the final steps are to teach it through training, check its progress through evaluation, and finally, ask it to generate new text.

### The Training Loop

This is the core engine where the model learns. The script iterates through the data thousands of times, making small adjustments to its internal parameters with each step to get better at predicting the next character.

```python
# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Every so often, check how we're doing
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 1. Get a new batch of data
    xb, yb = get_batch('train')

    # 2. Forward Pass: Get predictions and calculate the loss
    logits, loss = model(xb, yb)
    
    # 3. Backward Pass: Calculate gradients and update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

  * **Optimizer (`AdamW`)**: We use the AdamW optimizer, a popular and robust choice that effectively adjusts the learning rate for each of the model's millions of parameters, helping it converge to a good solution more efficiently.
  * **The Loop**: The process is simple yet powerful:
    1.  **Sample Data**: Get a fresh batch of inputs (`xb`) and corresponding targets (`yb`).
    2.  **Forward Pass**: Ask the model for its predictions (`logits`) and calculate how wrong it was (`loss`).
    3.  **Backward Pass**: Calculate the gradients (`loss.backward()`)—this determines in which direction to adjust the parameters to reduce the loss. Then, the optimizer takes a step in that direction (`optimizer.step()`).

### Evaluating Model Performance (`estimate_loss`)

How do we know if the model is actually learning or just memorizing the training data? By testing it on a **validation set**—data it has never seen during training. This is the job of the `estimate_loss` function. If the training loss goes down but the validation loss goes up, the model is **overfitting**.

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Switch to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Switch back to training mode
    return out
```

This function has three key components for a correct and efficient evaluation:

  * **`@torch.no_grad()`**: This decorator tells PyTorch not to calculate gradients during this process. Since we're not learning here, there's no need to store the intermediate values needed for backpropagation, which saves a significant amount of memory and computation.
  * **`model.eval()`**: This puts the model into evaluation mode, which is crucial because it disables layers like **Dropout**. If Dropout was active, it would randomly turn off neurons, leading to a noisy and inconsistent measurement of the model's true performance.
  * **`model.train()`**: After the evaluation is complete, this line switches the model back to training mode, re-enabling Dropout and other training-specific layers for the next learning iteration.

### Generating New Text

After thousands of training iterations, the model is ready to perform its ultimate task: writing new text. This is done through an **autoregressive** process where the model's own output is fed back to it as input for the next step.

```python
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
```

Here's the breakdown of the final command:

1.  **`context = torch.zeros(...)`**: We create a starting point for the generation—a single batch containing a single token with the ID `0` (which is typically a newline character).
2.  **`model.generate(...)`**: This method, defined inside our `BigramLanguageModel` class, takes the context and loops `max_new_tokens` times. In each loop, it predicts the next character, appends it to the sequence, and uses the now-longer sequence as the context for the next prediction.
3.  **`decode(...)`**: The `generate` method outputs a tensor of token IDs. The `decode` function uses our `itos` (integer-to-string) map to translate these numbers back into human-readable characters, revealing the text our model has written.

By repeating this process, the model writes new text, one character at a time, based on the context it has generated so far.

## 6. Model Optimization and Execution

Writing the model code is one thing; making it run fast is another. The provided script includes modern PyTorch features to significantly accelerate training and inference.

### `torch.compile(model)`

This is one of the most significant features of PyTorch 2.0 and later. It's a one-line change that can provide substantial speedups.

```python
model = BigramLanguageModel()
model = torch.compile(model).to(device) # Compile the model
````

  * **Concept (The "Why")**: Python is an incredibly flexible but relatively slow language. Every time your model runs, the Python interpreter has to process the code line by line, which creates a lot of overhead. `torch.compile()` acts as a **Just-In-Time (JIT) compiler**. It takes your PyTorch model, analyzes its structure, and converts the Python code into a highly optimized, low-level computation graph that can be executed much more efficiently on the GPU.

  * **How It Works (The "How")**:

    1.  **Graph Acquisition**: `torch.compile` runs your code once to trace the sequence of operations, creating a graph representation of your model's forward and backward passes.
    2.  **Graph Lowering**: This graph is then translated into a hardware-agnostic intermediate representation.
    3.  **Backend Compilation**: Finally, a backend compiler (like **Triton**) takes this representation and generates extremely fast, fused code (kernels) specifically for your GPU. It can merge multiple operations (e.g., a linear layer followed by a ReLU) into a single step, drastically reducing memory access and Python overhead.

The result is that your model performs the exact same calculations, but much faster.

### `torch.set_float32_matmul_precision('high')`

This line optimizes the most common operation in a Transformer: matrix multiplication.

```python
torch.set_float32_matmul_precision('high')
```

  * **Concept (The "Why")**: Modern NVIDIA GPUs (Ampere architecture and newer) contain specialized hardware units called **Tensor Cores**. These cores are designed to perform matrix multiplications at blistering speeds, but they achieve this by using lower-precision number formats. This setting allows PyTorch to leverage these cores.

  * **How It Works (The "How")**: By setting the precision to `'high'`, you're telling PyTorch that it can use the **TensorFloat-32 (TF32)** format for matrix multiplications. TF32 uses the same 8 bits for the exponent as standard 32-bit floating-point numbers (FP32) but reduces the mantissa (the precision part) from 23 bits to 10 bits.

This means you get a massive speedup from using the Tensor Cores with a precision that is almost as good as full FP32, making it a safe and effective optimization for most deep learning models.

-----

## 🏁 Conclusion

You have now walked through the entire lifecycle of a Transformer-based language model. We started with raw text, tokenized it into numbers, and built the core components—attention heads, feed-forward networks, and transformer blocks. We assembled these into a deep network, incorporated positional information, and defined a forward pass that transforms token IDs into predicted logits. Finally, we trained this model using an optimizer and a standard training loop and used it to generate new text.

This script, while simple, contains the fundamental concepts that power much larger models like GPT-3. By understanding how the data flows and transforms at each step, you've grasped the essence of the Transformer architecture.
