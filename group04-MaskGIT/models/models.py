import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1)
        self.ln = nn.LayerNorm(dim, eps=1e-12)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x):
        x_res = x # x_res keeps what to add to the residual connection
        x = self.ln(x) # 1. Layer norm
        attn_output, _ = self.self_attn(x, x, x) # 2. MHSA
        x = attn_output + x_res # 3. Add residual connection
        x_res = x # update the res accumulator
        x = self.ln(x) # 4. Layer norm
        x = self.mlp(x) # 5. MLP
        x += x_res # 6. Add residual con

        return x



class BidirectionalTransformer(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer, self).__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.num_codebook_vectors + 2, args.dim)  # Embedding layer for tokens
        self.pos_emb = nn.Parameter(torch.zeros(args.num_img_tok + 1, args.dim))  # Learnable positional embeddings
        self.blocks = nn.ModuleList([TransformerBlock(args.dim, args.hidden_dim) for _ in range(args.n_layers)])  # A list of Transformer Blocks
        self.fc = nn.Linear(args.dim, args.num_codebook_vectors + 2)  # Final linear layer for token prediction
        self.drop = nn.Dropout(p=0.1)  # Dropout layer

    def forward(self, x):
        # 1. get token emmbeddings
        x = self.tok_emb(x)
        # 2. add positional encoding
        t = x.size(1)        
        # First, we add the token embeddings and the positional embeddings together.
        # The [:, :t] is to ensure that we only use the first 't' positional embeddings,
        # where 't' is the sequence length of the current batch.
        x = self.drop(x + self.pos_emb[:t, :])
        # 3. Pass the embeddins thru transformer blocks
        for block in self.blocks:
            x = block(x)
        # 4. Get logits
        x = self.fc(x)
        return x

"""
## USAGE
args = {
    'dim': 768,  # Embedding dimensions
    'hidden_dim': 3072,  # Hidden dimensions
    'n_layers': 24,  # Number of transformer blocks/layers
    'num_codebook_vectors': 1024,  # vocab size
    'num_img_tok': 256  # Maximum sequence length
}
"""