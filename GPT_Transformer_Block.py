import torch
import torch.nn as nn

torch.manual_seed(1337)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)
n_emnd = 32 #embedding dimension #Increasing n_embd increases both training cost and inference time. 

char_to_idx = {ch: i for i, ch in enumerate(characters)}
idx_to_char = {i: ch for i, ch in enumerate(characters)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

#Train/Validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4
block_size = 8

def batch_creation(split):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """Single head of causal self-attention"""
    """The attention mechanism allows each token to attend to all previous tokens in the sequence, enabling the model to consider context when making predictions."""
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        #Causal mask (registered as buffer, not a parameter)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Attention scores (scaled dot-product)
        wei = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)  # (B, T, T)

        # Causal masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Normalize to probabilities
        wei = torch.softmax(wei, dim=-1)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of causal self-attention in parallel"""
    """The 32-dimensional embedding space is split into 4 parallel attention heads, each operating on an 8-dimensional subspace. The outputs from these heads are then concatenated to form the final output."""
    def __init__(self, num_heads, n_embd, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out
    
class FeedForward(nn.Module):
    """A simple feed-forward neural network for non-linear transformation"""
    """The feedforward network (MLP) allows each token to independently process and transform the contextual information it has gathered through self-attention."""
    """Given everything I now know, how should I transform it?"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    """A Transformer works by repeatedly alternating between attention (communication) and feedforward networks (computation) across multiple layers."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads=n_head, n_embd=n_embd, block_size=block_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

#The model first maps tokens into a shared embedding space where similar tokens can have similar vector representations, and then uses a linear head to convert those vectors into a probability distribution over the vocabulary.
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Token embeddings represent what a token is, positional embeddings represent where it is, and the language-model head converts the resulting context representation into next-token probabilities.
        # Each token directly predicts the next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_emnd)
        self.positional_embedding_table = nn.Embedding(block_size, n_emnd) #Positional embeddings inject order into the model.
        self.blocks = nn.Sequential(
            Block(n_emnd, n_head=4),
            Block(n_emnd, n_head=4),
            Block(n_emnd, n_head=4)
        ) # A stack of Transformer blocks, where each block performs one full “talk → think” cycle.
        self.lm_head = nn.Linear(n_emnd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_enb = self.token_embedding_table(idx)  # (B, T, C)
        pos_enb = self.positional_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = token_enb + pos_enb  # (B, T, C) #Positional embeddings inject order into the model.
        x = self.blocks(x) # multiple transformer blocks
        logits = self.lm_head(x) # (B, T, vocab_size) # language model head

        if target is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = target.view(B * T)
            loss = nn.CrossEntropyLoss()(logits, targets)

        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad() # we are not performing backpropagation or gradient computation during loss estimation that's why we use this decorator...
def estimate_loss(model, eval_iters=200):
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = batch_creation(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out

model = GPTModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 10500
eval_interval = 500

for step in range(max_iters):
    xb, yb = batch_creation("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"Step {step} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

start_token = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start_token, max_token=500)
print("\nGenerated Text:\n")
print(decode(generated[0].tolist()))
