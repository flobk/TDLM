import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reimplementation of the Transformer and pos embeddings the paper uses.
Reference (original code is in JAX): https://github.com/google-deepmind/md4/blob/main/md4/networks/sharded_transformer.py
"""

# RoPE Rotary Positional Embeddings
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes complex exponentials (cis) for RoPE. 
    Returns cos/sin tables to be used for rotation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()  
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """
    Applies the rotation to Queries (xq) and Keys (xk).
    Mathematically: (x + iy) * e^(i*theta)
    """
    cos = freqs_cos[None, :, None, :]
    sin = freqs_sin[None, :, None, :]
    # Split last dimension into real/imaginary parts for rotation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # Rotation formula: Re(a*b) = ac - bd, Im(a*b) = ad + bc
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    # Reassemble
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    """
    LLaMA-style MLP using SwiGLU (Swish-Gated Linear Unit).
    Conceptually split into Gate, Up-projection, and Down-projection.
    """
    def __init__(self, config):
        super().__init__()
        # Dimension scaling logic from LLaMA (2/3rds ratio)
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = 32 * ((hidden_dim + 31) // 32) # Round to multiple of 32
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False) # Down
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Up
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: (Swish(Gate) * Up) -> Down
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    """
    Standard Multi-Head Attention with RoPE injected into Q and K.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        
        xq = self.wq(x).view(B, T, self.n_head, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_head, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_head, self.head_dim)
        
        # Inject position info via rotation before attention calculation
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # Reshape to [B, Heads, T, Dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash Attention (scaled_dot_product_attention)
        out = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.attn_dropout.p if self.training else 0.0)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(out))


class TransformerBlock(nn.Module):
    """
    DiT / LLaMA Style Block.
    Uses Adaptive Layer Norm (AdaLN) to inject the timestep 't' into the network.
    The time embedding shifts and scales the normalized features.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        
        # affine=False because we learn the shift/scale dynamically from t
        self.attention_norm = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-5)
        
        # AdaLN Modulation: Regresses 6 parameters from time embedding
        # 1. Shift Attention, 2. Scale Attention, 3. Gate Attention
        # 4. Shift MLP,       5. Scale MLP,       6. Gate MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )
        
        # Zero-init gating ensures the block acts as Identity function at initialization
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, t_emb, freqs_cos, freqs_sin):
        # Regress modulation parameters [B, 6*D] -> 6 chunks of [B, D]
        chunks = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        shift_att, scale_att, gate_att = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]
        
        # Attention Block with AdaLN
        # x = x + gate * Attention( (x-mean)/std * (1+scale) + shift )
        x_norm = self.modulate(self.attention_norm(x), shift_att, scale_att)
        x = x + gate_att.unsqueeze(1) * self.attention(x_norm, freqs_cos, freqs_sin)
        
        # MLP Block with AdaLN
        x_norm = self.modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.feed_forward(x_norm)
        
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)