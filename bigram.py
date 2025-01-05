import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 4000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)
eval_iters = 200
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
    """generate a small batch of data of inputs x and targets y"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Using torch.no_grad() to disable gradient computation
@torch.no_grad()
def estimate_loss():
    """compute the loss on both train and val sets on eval_iters batches"""
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


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a
        # lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def poly_val(self, x, coeffs):
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

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # 4) Compute the Shannon entropy of these probabilities
            #    Use clamp(...) to avoid log(0). Keep dim=-1 for the sum,
            #    but keepdim=True so we can broadcast easily later.
            entropy = -torch.sum(
                probs * torch.log(probs.clamp_min(1e-9)), dim=-1, keepdim=True
            )  # (B, 1)

            # 5) Define polynomial fit coefficients for adaptive temperature
            #    (Same as in the paper’s JAX example: [-0.037, 0.481, -2.3, 4.917, -1.791]).
            poly_fit = torch.tensor(
                [-0.037, 0.481, -2.3, 4.917, -1.791],
                dtype=logits.dtype,
                device=logits.device,
            )

            # 6) Compute the temperature scaling factor beta = 1/theta
            #    If the entropy <= 0.5, leave beta=1.0 to avoid overcorrection
            #       on low-entropy heads.
            #    Otherwise, evaluate the polynomial, clamp it at min=1.0 to
            #       never increase entropy.
            beta = torch.where(
                entropy > 0.5,
                self.poly_val(entropy, poly_fit).clamp_min(1.0),
                torch.tensor(1.0, dtype=logits.dtype, device=logits.device),
            )  # (B, 1)

            # 7) Rescale logits by beta
            scaled_logits = logits * beta  # broadcast multiply (B, C) * (B, 1)

            # 8) Re-softmax with the scaled logits
            adaptive_probs = F.softmax(scaled_logits, dim=-1)  # (B, C)

            # 9) Sample from the distribution
            idx_next = torch.multinomial(
                adaptive_probs, num_samples=1
            )  # (B, 1)

            # 10) Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
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
