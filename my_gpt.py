"""my_gpt.py"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000  # TODO 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
adaptive_temperature = False
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def poly_val(x, coeffs):
    """
    Similar to https://arxiv.org/pdf/2410.01104.pdf, we use a polynomial to scale the logits.
    Evaluate a polynomial with given coeffs (highest degree first).
    E.g., if coeffs = [a, b, c, d, e], then poly_val(x, coeffs) = a*x^4 + b*x^3 + c*x^2 + d*x + e.
    """
    # x is shape (B, 1), so each x**k is broadcast accordingly
    return (
        coeffs[0] * x**4
        + coeffs[1] * x**3
        + coeffs[2] * x**2
        + coeffs[3] * x
        + coeffs[4]
    )


def adapt_temperature(
    logits: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """
    Adapt the temperature of the softmax distribution based on the entropy of the distribution.
    https://arxiv.org/pdf/2410.01104.pdf
    """
    # 0) Compute the Shannon entropy of these probabilities
    #    Use clamp(...) to avoid log(0). Keep dim=-1 for the sum,
    #    but keepdim=True so we can broadcast easily later.
    entropy = -torch.sum(
        probs * torch.log(probs.clamp_min(1e-9)),
        dim=-1,
        keepdim=True,
    )  # (B, 1)

    # 1) Define polynomial fit coefficients for adaptive temperature
    #    (Same as in the paper’s example: [-0.037, 0.481, -2.3, 4.917, -1.791]).
    poly_fit = torch.tensor(
        [-0.037, 0.481, -2.3, 4.917, -1.791],
        dtype=logits.dtype,
        device=device,
    )

    # 2) Compute the temperature scaling factor beta = 1/theta
    #    If the entropy <= 0.5, leave beta=1.0 to avoid overcorrection
    #       on low-entropy heads.
    #    Otherwise, evaluate the polynomial, clamp it at min=1.0 to
    #       never increase entropy.
    beta = torch.where(
        entropy > 0.5,
        poly_val(entropy, poly_fit).clamp_min(1.0),
        torch.tensor(1.0, dtype=logits.dtype, device=device),
    )  # (B, 1)

    # 3) Rescale logits by beta
    scaled_logits = logits * beta  # broadcast multiply (B, C) * (B, 1)

    # 4) Re-softmax with the scaled logits
    probs = F.softmax(scaled_logits, dim=-1)  # (B, C)
    return probs


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, adaptive_temperature):
        super().__init__()
        self.adaptive_temperature = adaptive_temperature
        # token embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):  # [B, T]
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C=n_embd)
        # this is because the tokens have indexes 0, 1, ... T-1 where T is the
        # block size.
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        # compute the logits from x
        logits = self.lm_head(x)  # (B,T,V=vocab_size)
        if targets is None:
            loss = None
        else:
            # compute the loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
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
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            if self.adaptive_temperature:
                probs = adapt_temperature(logits, probs)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = MyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    adaptive_temperature=adaptive_temperature,
)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
torch.manual_seed(1337)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
