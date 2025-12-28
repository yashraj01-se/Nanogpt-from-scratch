import torch
import torch.nn as nn

#DEVICE SETUP (CUDA) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)

with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)

char_to_idx = {ch: i for i, ch in enumerate(characters)}
idx_to_char = {i: ch for i, ch in enumerate(characters)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long, device=device)

# TRAIN / VALIDATION SPLIT
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# HYPERPARAMETERS
n_emnd = 384
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
dropout_rate = 0.2
max_iters = 5000
eval_interval = 500

#  BATCH CREATION 
def batch_creation(split):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,), device=device)
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ATTENTION HEAD 
class Head(nn.Module):
    """Single head of causal self-attention"""
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v

# MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    """Multiple heads of causal self-attention in parallel"""
    def __init__(self, num_heads, n_embd, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)
        return out

# FEEDFORWARD
class FeedForward(nn.Module):
    """Feed-forward network for token-wise computation"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

# TRANSFORMER BLOCK
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# GPT MODEL 
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emnd)
        self.positional_embedding_table = nn.Embedding(block_size, n_emnd)

        self.blocks = nn.Sequential(
            *(Block(n_emnd, n_head) for _ in range(n_layer))
        )

        self.ln_f = nn.LayerNorm(n_emnd)
        self.lm_head = nn.Linear(n_emnd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_enb = self.token_embedding_table(idx)
        pos_enb = self.positional_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = token_enb + pos_enb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(B * T, vocab_size)
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

# LOSS ESTIMATION 
@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = batch_creation(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# TRAINING
model = GPTModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(max_iters):
    xb, yb = batch_creation("train")
    _, loss = model(xb, yb)

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

#  GENERATION 
start_token = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(start_token, max_token=500)

print("\nGenerated Text:\n")
print(decode(generated[0].tolist()))

# Output Example:
#Step 0 | Train Loss: 3.5415 | Val Loss: 3.5610
# Step 500 | Train Loss: 1.8823 | Val Loss: 1.9970
# Step 1000 | Train Loss: 1.5379 | Val Loss: 1.7130
# Step 1500 | Train Loss: 1.3951 | Val Loss: 1.6090
# Step 2000 | Train Loss: 1.3140 | Val Loss: 1.5528
# Step 2500 | Train Loss: 1.2499 | Val Loss: 1.5150
# Step 3000 | Train Loss: 1.2075 | Val Loss: 1.5016
# Step 3500 | Train Loss: 1.1626 | Val Loss: 1.4903
# Step 4000 | Train Loss: 1.1214 | Val Loss: 1.4861
# Step 4500 | Train Loss: 1.0850 | Val Loss: 1.4866

# Generated Text:


# KLEWIS XV:
# By journeys, told The Duke of York now, the endity hid:
# And more of the winds we did I am bound
# Than it shamefully else: one gase my fame,
# If I would you fight from a bear my perish,
# Offiance death mercy with some that wings?
# I have moe the passips you there more desire to offence.
# Out, my son, shall I with him my heart with on
# His mahtmery ward in abusenes: being cled divorce to-day.

# POLIXENES:
# The fight of my good formity, and
# thy walcome: or be my son, and used his life, deserves,