import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from transformer_impl import TransformerBlock
from transformer_impl import precompute_freqs_cis

"""
Reimplementation of the MD4 discrete diffusion model. 
Refernce (original code is in JAX): https://github.com/google-deepmind/md4/blob/main/md4/models/diffusion/md4.py

The reimplemenation is limited to only use:
- cosine schedule as the alpha schedule
- ancestral sampling
- continuous time sampling
"""

class TimestepEmbedder(nn.Module):
    """
    Maps a scalar timestep t to a high-dimensional vector.
    Uses Fourier features followed by an MLP to allow the network to 
    distinguish high-frequency details in the noise schedule.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size * 4),
            nn.SiLU(), # They call it "swish" in their JAX code
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t = t * 1000.0
        half_dim = self.frequency_embedding_size // 2
        # Fourier feature calculation: sin(w*t), cos(w*t)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)


class MD4Config:
    """
    Specific parameter config used for this run.
    (Also includes the training params).
    """
    def __init__(self, vocab_size):
        # Model params
        self.vocab_size = vocab_size # (90 see our training data)
        self.block_size = 128 # (sequence length)
        self.n_embd = 512      
        self.n_head = 8     
        self.n_layer = 8
        self.dropout = 0.1
        # Training params
        self.max_steps = 50000
        self.batch_size = 128
        self.learning_rate = 2e-4
        self.min_lr = 1e-5           # Decay to 5%
        self.warmup_steps = 1000


class MD4(nn.Module):
    """ 
    Masked Discrete Diffusion with DiT architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_token_id = config.vocab_size 
        self.internal_vocab_size = config.vocab_size + 1  # Add the MASK state
        
        self.token_emb = nn.Embedding(self.internal_vocab_size, config.n_embd)
        self.time_emb = TimestepEmbedder(config.n_embd, frequency_embedding_size=config.n_embd)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final Norm is also modulated by time
        self.final_norm = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-5)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )
        
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

        self.eps = 1e-4 # Buffer parameter for alpha calculation

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # Diffusion Schedule
    def get_alpha(self, t):
        """
        Cosine Schedule: alpha(t) = 1 - cos(pi/2 * (1-t)).
        Concept: Alpha represents the probability of a token being 'Clean' (Unmasked).
        t=0 -> Alpha=1 (All Clean)
        t=1 -> Alpha=0 (All Masked)

        Purpose of eps:
        Squashes alpha into range [eps, 1-eps], prevents '1-alpha' from becoming 0 (log(0) issues if converting to log-SNR)
        """
        t = torch.clamp(t, 0.0, 1.0)
        # Raw cosine schedule
        alpha_raw = 1.0 - torch.cos(math.pi / 2.0 * (1.0 - t))
        # Add eps as they do
        return (1.0 - 2 * self.eps) * alpha_raw + self.eps

    def get_dalpha(self, t):
        """
        Derivative of alpha with respect to t: d(alpha)/dt.
        Used to calculate how 'fast' information is destroyed/created at time t.
        
        Since we scaled alpha by (1 - 2*eps), we must scale the derivative as well.
        """
        u = math.pi / 2.0 * (1.0 - t)
        dalpha_raw = (math.pi / 2.0) * torch.sin(u)
        return (1.0 - 2 * self.eps) * dalpha_raw

    def get_loss_weight(self, t):
        """
        Calculates the VLB (Variational Lower Bound) weight.
        Weight = |d_alpha/dt| / (1 - alpha)
        This ensures the loss approximates the likelihood integral correctly.
        """
        d_alpha = self.get_dalpha(t)
        one_minus_alpha = 1.0 - self.get_alpha(t)
        return d_alpha / one_minus_alpha


    def forward(self, x, t):
        B, T = x.shape
        x_emb = self.token_emb(x)
        t_emb = self.time_emb(t)

        freqs_cos, freqs_sin = precompute_freqs_cis(self.config.n_embd // self.config.n_head, T)
        freqs_cos, freqs_sin = freqs_cos.to(x.device), freqs_sin.to(x.device)

        for block in self.blocks:
            x_emb = block(x_emb, t_emb, freqs_cos, freqs_sin)

        shift_out, scale_out = self.final_adaLN(t_emb).chunk(2, dim=1)
        x_out = self.final_norm(x_emb) * (1 + scale_out.unsqueeze(1)) + shift_out.unsqueeze(1)

        return self.head(x_out)

    def compute_loss(self, x0):
        """
        Training Step (Continuous Time ELBO):
        1. Sample time t.
        2. Mask input x0 based on schedule alpha(t).
        3. Predict original tokens x0.
        4. Compute weighted loss (VLB).
        Total ELBO = loss_diff + loss_prior + loss_recon
        """
        B, T = x0.shape
        device = x0.device
        
        # Sample time and create noised input by masking
        t = torch.rand(B, device=device)
        alpha = self.get_alpha(t)[:, None]
        mask_mask = torch.rand(B, T, device=device) > alpha
        x_t = x0.clone()
        x_t[mask_mask] = self.mask_token_id
        
        # Forward pass and compute loss
        logits = self.forward(x_t, t)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), x0.view(-1), reduction='none')
        loss = loss.view(B, T)
        
        # Compute weighted loss on masked positions (diffusion loss)
        masked_loss = (loss * mask_mask.float()).sum(dim=1)
        weight = self.get_loss_weight(t)
        loss_diff = (weight * masked_loss).mean()
        
        # Reconstruction loss: gap at t=0 where alpha is not exactly 1
        # loss_recon = sequence_length * (1 - alpha(0)) * log(vocab_size)
        alpha_0 = self.get_alpha(torch.tensor(0.0, device=device))
        loss_recon = T * (1.0 - alpha_0) * math.log(self.config.vocab_size)
        
        # Prior loss: KL divergence at t=1 (0 because we are doing masked diffusion)
        loss_prior = 0.0
        
        # Total ELBO loss
        loss_total = loss_diff + loss_prior + loss_recon
        
        return loss_total

    @torch.no_grad()
    def generate(self, seq_len=256, steps=64):
        """
        Ancestral Sampling (Reverse Process):
        Iteratively unmask tokens from t=1 (All Masked) to t=0 (All Clean).
        """
        self.eval()
        device = self.head.weight.device
        # start with a fully masked sequence
        x = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        for i in range(steps):
            # One ancestral sampling step: update each mask with probability given by diffusion schedule
            t_curr = 1.0 - (i / steps)
            t_next = 1.0 - ((i + 1) / steps)

            alpha_curr = self.get_alpha(torch.tensor([t_curr], device=device)).item()
            alpha_next = self.get_alpha(torch.tensor([t_next], device=device)).item()
            
            t_tensor = torch.full((1,), t_curr, device=device)
            logits = self.forward(x, t_tensor)
            probs_x0 = F.softmax(logits, dim=-1)

            if alpha_curr >= 0.999:
                unmask_prob = 1.0
            else:
                unmask_prob = (alpha_next - alpha_curr) / (1.0 - alpha_curr)
            unmask_prob = max(0.0, min(1.0, unmask_prob))

            # Build categorical distribution over [all tokens | keep mask]
            probs_outcome = probs_x0 * unmask_prob
            prob_keep_mask = torch.tensor(1.0 - unmask_prob, device=device).view(1, 1, 1).expand(1, seq_len, 1)
            full_probs = torch.cat([probs_outcome, prob_keep_mask], dim=-1)
        
            # Sample outcome
            sampled_indices = torch.multinomial(full_probs.view(-1, self.config.vocab_size + 1), 1).view(1, seq_len)

            # If index < vocab_size: reveal token; else: keep mask
            new_tokens = sampled_indices
            kept_mask = (sampled_indices == self.config.vocab_size)
            
            is_currently_masked = (x == self.mask_token_id)
            update_mask = is_currently_masked & (~kept_mask)
            x = torch.where(update_mask, new_tokens, x)
        
        self.train()
        return x[0].tolist()

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Helper method.
        Splits parameters into weight_decay (Linear/Embeddings) 
        and no_weight_decay (Biases/LayerNorms/Scales).
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Optimizer using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)
        return optimizer
