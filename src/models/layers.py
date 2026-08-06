import torch 
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter gamma, initialized to 1s
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim)
        # Calculate mean of squared activations along the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Multiply by reciprocal square root (1 / sqrt(variance + eps))
        x_normed = x * torch.rsqrt(variance + self.eps)
        # Scale element-wise by learnable weight vector
        return self.weight * x_normed    
    
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * out_features, bias=bias)
    
    def forward(self, x):
        x = self.w12(x)
        gate, value = x.chunk(2, dim=-1)
        return value * F.silu(gate)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.act = SwiGLU(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
    def forward(self, x):
        x = self.act(x)
        x = self.fc2(x)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x shape: (batch, num_heads, seq_len, head_dim)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

