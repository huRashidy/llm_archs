from typing import Optional

import torch 
import math
from torch import nn
from torch.nn import functional as F
from .layers import RMSNorm, MLP, RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor , n_rep: int) -> torch.Tensor:
    """
    Repeat the key and value tensors for GQA (Grouped Query Attention).
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        n_rep (int): Number of repetitions for each head.
    Returns:
        torch.Tensor: Repeated tensor of shape (batch_size, num_heads * n_rep, seq_len, head_dim).
    """
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim).reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)

class Attention(
    nn.Module,
):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        vocab_size: int = 50257,
        batch_first: bool = True,
        use_GQA: bool = False,
        num_kv_heads:Optional[int] = None,
        clm: bool = True,
        rope: bool= True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.clm = clm
        self.rope = rope
        self.use_GQA = use_GQA
        self.head_dim = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv_head = num_heads // self.num_kv_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if self.rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        else: 
            self.rotary_emb = None

    def forward(self, x , clm=True):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            cos, sin = self.rotary_emb(q, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if not self.use_GQA and self.num_kv_heads == self.num_heads:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        elif self.use_GQA or self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.num_queries_per_kv_head)
            v = repeat_kv(v, self.num_queries_per_kv_head)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            raise ValueError("Invalid configuration for GQA and number of KV heads.")
        if clm:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=attn_weights.device, dtype=torch.bool))
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)
        context = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(context)

        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, mlp_ratio=4.0, dropout=0.0, bias=False, rope=True , parallel_residual=False):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, num_kv_heads=num_kv_heads, dropout=dropout, bias=bias, rope=rope)
        self.norm2 = RMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=hidden_dim)
        self.parallel_residual = parallel_residual

    def forward(self, x):
        if not self.parallel_residual:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out = self.attn(self.norm1(x))
            mlp_out = self.mlp(self.norm2(x))
            x = x + attn_out + mlp_out
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_kv_heads, num_heads, vocab_size = 50257, mlp_ratio=4.0, dropout=0.0, bias=False, rope=True , parallel_residual=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads=num_kv_heads, mlp_ratio=mlp_ratio, dropout=dropout, bias=bias, rope=rope, parallel_residual=parallel_residual)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

