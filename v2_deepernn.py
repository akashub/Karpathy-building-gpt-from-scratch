# Adding Block Feedforward layer, Add and Norm Skip layers after Multi-Head Self-Attention layer

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # changed from 1e-2 to 1e-3 because with self-attention the model was overfitting very quickly and therefore we reduce the learning rate to make the training more stable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Using CUDA")
    torch.cuda.empty_cache()
    print("Cache was cleared")
else:
    print("Using CPU")
eval_iters = 200
n_embd = 256 # number of embedding dimensions -> 384/6 = 64 dimensions per head if we have 6 heads
dropout = 0.2
n_head = 8 # number of heads in multi-head attention
assert n_embd % n_head == 0 # make sure that the number of embedding dimensions is divisible by the number of heads
n_layer = 4 # number of layers in the transformer (number of blocks)
torch.set_float32_matmul_precision('high')
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('tinyShakeSpeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
# @torch.no_grad() is a decorator that will disable gradient tracking for the entire function -> we do this because we aren't storing any intermediate results for backpropagation when we are just evaluating the loss and not training anything and therefore it saves a lot of memory and computations
def estimate_loss():
    out = {}
    model.eval() # we start with eval and then reset back to train beacause we don't want to update any parameters of the model when we are just evaluating the loss and therefore we set the model to eval mode which will disable dropout and batchnorm layers if any are present in the model
    # "I’m not training right now, I just want to measure how well the model is doing."
    # This ensures:
        # Predictions are deterministic (no random dropout).
        # Validation loss is comparable across evaluations.
        # The loss reflects the model’s “true” ability, not training noise.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # start with a tensor of zeros of size eval_iters = 200
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # reset back to training mode
    return out

class Head(nn.Module):
    """ one head of self attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # We set bias=False because we don't need bias in the linear layers that produce keys, queries, and values. The bias term would just add a constant offset to all the vectors, which doesn't provide any useful information for the attention mechanism. And since mean centering i.e mean = 0 and var = 1 is done during initialisation of weights, adding a bias term wouldn't help and would increase the overhead of the model unnecessarily
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # tril is not a parameter of the model, it's a constant that we use in the forward pass to mask out future tokens. Therefore, we register it as a buffer so that it gets moved to the appropriate device (CPU or GPU) along with the model, but it won't be updated during training.
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,16)
        q = self.query(x) # (B,T,16)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) --- C is head_size here
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) -> Decoder block --- self.tril[:T, :T] == 0 means that we are only considering the first T tokens in the sequence. This is important because during generation, the sequence length T can be less than the block_size. By slicing the tril matrix to [:T, :T], we ensure that the mask is always the correct size for the current input sequence length. If we didn't do this, we might end up with a mask that is too large, which would lead to incorrect masking and potentially allow the model to attend to future tokens that it shouldn't be able to see. We equate it with 0 because tril has 1s in the lower triangle and 0s elsewhere, so this condition identifies the positions that need to be masked out. We replace those positions with -inf to ensure that after applying softmax, those positions will have zero attention weight.

        wei = F.softmax(wei, dim=-1) #(B, T, T) -- F is the functional module in PyTorch which contains functions that are stateless and can be used directly. Here we use F.softmax to convert the attention scores into probabilities. The dim=-1 argument specifies that the softmax should be applied along the last dimension of the tensor, which corresponds to the different tokens in the sequence. This means that for each token in the sequence, we are calculating how much attention it should pay to every other token, and the softmax ensures that these attention weights sum to 1.
        # perform the weighted aggregation of the values
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, 16)
        out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # create a list of heads
        self.proj = nn.Linear(n_embd, n_embd) # projection layer to combine the outputs of the multiple heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of the heads along the embedding dimension
        out = self.dropout(self.proj(out)) # project back to n_embd dimensions
        return out

class Feedforward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # as per the paper we increase the dimension to 4 times the embedding dimension (512 -> 2048) (up projection)
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # Projection layer back to n_embd dimensions, so that we can add the input to the output (residual connection) --- 2048 -> 512 (down projection)
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation -> intersperse these two operations multiple times """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention layer
        self.ffwd = Feedforward(n_embd) # feedforward layer
        self.ln1 = nn.LayerNorm(n_embd) # layernorm 1
        self.ln2 = nn.LayerNorm(n_embd) # layernorm 2

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # apply layernorm 1, then self-attention, then add skip connection/ residual connection
        x = x + self.ffwd(self.ln2(x)) # apply layernorm 2, then feedforward, then add skip connection/ residual connection
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd = number of embedding dimensions -> we introduce this to have an intermidiate representation of the input tokens before projecting them to the output vocabulary space
        # self.sa_head = Head(n_embd) # self-attention head
        # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//n_head) # multi-head self-attention -> we use 4 heads and each head has a dimension of n_embd/4 so that when we concatenate the outputs of the 4 heads we get back to n_embd dimensions -> 4 heads of 8 dimensions each = 32 dimensions
        # self.ffwd = Feedforward(n_embd)
        # self.Blocks = nn.Sequential(
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     nn.LayerNorm(n_embd) # final layernorm at the end of all the blocks (before the output layer)
        # )
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head = n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd) # final layernorm at the end of all the blocks (before the output layer)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head -> projects the n_embd dimensional embeddings to the vocab_size dimensional logits for each token
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # positional embeddings -> we add this to give the model a sense of order of the tokens in the sequence
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm at the end of all the blocks (before the output layer)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        C = self.token_embedding_table.embedding_dim
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.positional_embedding_table(torch.arange(0, T, device=device)) # (T,C)
        x = tok_embd + pos_embd # (B,T,C) -> we add
        x = self.blocks(x)
        x = cp.checkpoint_sequential(self.blocks, segments=2, input=x, use_reentrant=False) # (B, T, C) -> use checkpointing to save memory
        # cp.checkpoint_sequential is a function in PyTorch that allows you to trade compute for memory during the training of deep neural networks. It does this by saving only a subset of the intermediate activations during the forward pass and recomputing them during the backward pass. This can be particularly useful when training very deep networks or when working with limited GPU memory.
        # segments=2 means that the sequence of layers in self.blocks will be divided into 2 segments for checkpointing. This means that during the forward pass, only the outputs of these 2 segments will be saved, and the intermediate activations within each segment will be discarded to save memory. During the backward pass, the discarded activations will be recomputed as needed.
        # use_reentrant=False is an argument that controls how the recomputation of activations is handled during the backward pass. When set to False, it uses a non-reentrant approach, which can be more memory efficient in some cases. However, it may not work correctly with certain types of layers or operations that require reentrant behavior. Setting it to False is often a good default choice unless you encounter issues that require reentrant behavior.


        # x = self.sa_heads(x) # apply self-attention head
        # x = self.ffwd(x) # (B, T, C)
        # x = self.blocks(x)
        x = self.ln_f(x) # final layernorm
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model = torch.compile(model).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # 10 M parameter model on roughly 300,000 tokens of training data -> this is a small model

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))