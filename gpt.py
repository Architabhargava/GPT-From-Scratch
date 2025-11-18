import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # GPT is more complex than the bigram model, so we train it for more iterations. (Also the learning rate is lower, so it needs more iterations to converge.)
eval_interval = 500 # we evaluate the model every 500 iterations because it takes longer to train and we don't need to evaluate it as frequently.
learning_rate = 3e-4 # a smaller learning rate for GPT because it is a more complex model and can easily overfit with a high learning rate.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # why does bigram and gpt both use 200 eval iters? Because it's a good balance between getting a reliable estimate of the loss and not taking too long to compute.
# When we evaluate the model, we want to get a good estimate of the loss. We do this by averaging the loss over multiple batches. eval_iters is the number of batches we use to compute this average.       
n_embd = 384 # embedding dimension. it is equal to 32 in the bigram model, but we increase it to 384 for GPT to give the model more capacity.
n_head = 6 # number of heads in the multi-head attention mechanism. head_size = n_embd // n_head = 384 // 6 = 64
n_layer = 6 # number of transformer blocks, here, decoder blocks because GPT is a decoder-only transformer.
dropout = 0.2 # Every forward-backward pass, we randomly zero out 20% of the neurons in the model to prevent overfitting.
# ------------

torch.manual_seed(1337)

with open('dataset/input.txt', 'r', encoding='utf-8') as f:
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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # nn.Linear is a fully connected layer. It takes input of size n_embd and outputs size head_size.
        self.query = nn.Linear(n_embd, head_size, bias=False) # we are not taking bias because we want the attention to be based purely on the input embeddings and not have any additional bias term.
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
# Multi heads allow the model to learn from multiple "perspectives": maybe one head pays attnetion to direct neighbors, another to verbs, another to nouns, etc.
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Creates a list of Head modules, each with its own parameters. Each head performs independent attention computations.
        self.proj = nn.Linear(head_size * num_heads, n_embd) # Linear layer to project the concatenated outputs of all heads back to the original embedding dimension. head_size * num_heads ---> n_embd
        self.dropout = nn.Dropout(dropout) # Dropout layer (regularizer) randomly sets a fraction of input units (neurons) to zero during training to prevent overfitting. This ensures that the model does not rely too heavily on any particular neuron and encourages it to learn more robust features.

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the output of each head into a list along the last dimension(feature/embedding dimension)
        out = self.dropout(self.proj(out)) # project the concatenated output back to n_embd dimension and apply dropout for regularization.
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
# The whole idea of this layer is to apply a non-linear transformation to each position in the sequence independently and identically.
# The tokens basically think on the data that they've aggregated from the attention layer individually.
    def __init__(self, n_embd): # n_embd is the size of the token embeddings, i.e., how many features each token vector has. n_embd is also the input and output dimension of the FeedForward layer.
        super().__init__()
        self.net = nn.Sequential( # a container that allows us to stack layers together in a sequential manner.
            nn.Linear(n_embd, 4 * n_embd), # first linear layer expands the embedding dimension from n_embd to 4*n_embd. This expansion allows the model to learn more complex representations by projecting the input into a higher-dimensional space.
            # We chose 4 because the Attention Is All You Need paper used this expansion factor and it has been found to work well in practice.
            nn.ReLU(), # adding non-linearity to the model. Without it, the FeedForward layer would just be a linear transformation, which limits the model's capacity to learn complex patterns.
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # 
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (MultiHead Self Attention) followed by computation (Feed Forward) """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension = 384, n_head: the number of heads we'd like = 6
        super().__init__()
        head_size = n_embd // n_head # in our case, 384/6 = 64. Each head will have an embedding dimension of 64. 
        self.sa = MultiHeadAttention(n_head, head_size) # Creates a multi-head attention layer with n_head(6) heads, each of size head_size(64).
        self.ffwd = FeedForward(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd) # Layer normalization layer that normalizes the input across the feature dimension (n_embd). 
        # It normalizes the inputs to have zero mean and unit variance, which helps stabilize and accelerate training. It also helps with the vanishing/exploding gradient problem.
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # The addition here is a residual connection. We need it to help with the vanishing gradient problem. It allows the gradient to flow directly through the network without passing through non-linearities that could diminish it.
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding layer from vocab_size to n_embd because each token is represented by a vector of n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding layer from block_size to n_embd because each position in the block (from 0 to block_size-1) is represented by a vector of n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Linear Layer from n_embd to vocab_size which outputs logits for each token in the vocabulary

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) where C = n_embd. 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) where integers 0 to T-1 are mapped to n_embd dimensional vectors. integers here mean positions in the block
        x = tok_emb + pos_emb
        # How does the addition work here? Broadcasting. tok_emb is (B,T,C) and pos_emb is (T,C). pos_emb will be automatically broadcast over the batch dimension—that is, for each element in the batch, the same positional embedding is added.
        #Effectively, pos_emb is treated as (1, T, C), and then broadcast/repeated across the B batch dimension.
        
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
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
            idx_cond = idx[:, -block_size:] # Done because the model can only attend to the last block_size tokens, so we can ignore the earlier tokens. We never pass in a sequence longer than block_size.
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

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))