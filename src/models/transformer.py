from typing import Optional, Tuple

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
        num_kv_heads: Optional[int] = None,
        clm: bool = True,
        rope: bool = True
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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        clm: bool = True
    ):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            cos, sin = self.rotary_emb(q, seq_len, start_pos=start_pos)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        total_seq_len = k.shape[2]

        if not self.use_GQA and self.num_kv_heads == self.num_heads:
            k_rep = k
            v_rep = v
        elif self.use_GQA or self.num_kv_heads < self.num_heads:
            k_rep = repeat_kv(k, self.num_queries_per_kv_head)
            v_rep = repeat_kv(v, self.num_queries_per_kv_head)
        else:
            raise ValueError("Invalid configuration for GQA and number of KV heads.")

        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if clm and seq_len > 1:
            mask = torch.tril(torch.ones((seq_len, total_seq_len), device=attn_weights.device, dtype=torch.bool), diagonal=start_pos)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v_rep)
        context = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(context)

        if use_cache:
            return attn_output, present_kv
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, mlp_ratio=4.0, dropout=0.0, bias=False, rope=True, parallel_residual=False):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, num_kv_heads=num_kv_heads, dropout=dropout, bias=bias, rope=rope)
        self.norm2 = RMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=hidden_dim)
        self.parallel_residual = parallel_residual

    def forward(self, x, past_kv=None, use_cache=False, start_pos=0):
        if use_cache:
            attn_out, present_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=True, start_pos=start_pos)
        else:
            attn_out = self.attn(self.norm1(x), past_kv=past_kv, use_cache=False, start_pos=start_pos)
            present_kv = None

        if not self.parallel_residual:
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        else:
            mlp_out = self.mlp(self.norm2(x))
            x = x + attn_out + mlp_out

        if use_cache:
            return x, present_kv
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_kv_heads, num_heads, vocab_size=50257, mlp_ratio=4.0, dropout=0.0, bias=False, rope=True, parallel_residual=False):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads=num_kv_heads, mlp_ratio=mlp_ratio, dropout=dropout, bias=bias, rope=rope, parallel_residual=parallel_residual)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, past_key_values=None, use_cache=False, start_pos=0):
        x = self.embed_tokens(x)
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present_kv = layer(x, past_kv=past_kv, use_cache=True, start_pos=start_pos)
                present_key_values.append(present_kv)
            else:
                x = layer(x, past_kv=past_kv, use_cache=False, start_pos=start_pos)

        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, present_key_values
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        if not use_cache:
            for _ in range(max_new_tokens):
                logits = self(idx)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        else:
            past_key_values = None
            start_pos = 0
            curr_idx = idx

            for _ in range(max_new_tokens):
                logits, past_key_values = self(curr_idx, past_key_values=past_key_values, use_cache=True, start_pos=start_pos)
                start_pos += curr_idx.shape[1]

                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                curr_idx = idx_next

            return idx

