"""DINO attention K/V extraction and projection for AttnScaf."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderKVExtractor(nn.Module):
    """Capture native K/V tensors from selected timm/DINO attention layers."""
    def __init__(self, encoder, layer_indices):
        super().__init__()
        self.encoder = encoder
        self.layer_indices = list(layer_indices)
        self._kv_cache = {}
        self._hooks = []
        blocks = encoder.blocks
        for idx in self.layer_indices:
            if idx < 0 or idx >= len(blocks):
                raise ValueError(f"DINO layer {idx + 1} is outside 1..{len(blocks)}")
            self._hooks.append(blocks[idx].attn.register_forward_hook(self._hook(idx)))

    def _get_prefix_token_count(self, sequence_length=None):
        num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", None)
        if num_prefix_tokens is None:
            has_cls_token = getattr(self.encoder, "cls_token", None) is not None
            num_register_tokens = int(getattr(self.encoder, "num_register_tokens", 0))
            num_prefix_tokens = int(has_cls_token) + num_register_tokens
        else:
            num_prefix_tokens = int(num_prefix_tokens)

        # DINOv2 does not consistently expose num_prefix_tokens. Validate the
        # metadata against the square patch grid and infer it when necessary.
        if sequence_length is not None:
            num_patch_tokens = sequence_length - num_prefix_tokens
            patch_side = math.isqrt(num_patch_tokens) if num_patch_tokens >= 0 else -1
            if patch_side < 0 or patch_side * patch_side != num_patch_tokens:
                for candidate in range(min(32, sequence_length)):
                    num_patch_tokens = sequence_length - candidate
                    patch_side = math.isqrt(num_patch_tokens)
                    if patch_side * patch_side == num_patch_tokens:
                        num_prefix_tokens = candidate
                        break
                else:
                    raise ValueError(
                        f"Cannot infer DINO prefix tokens from sequence length {sequence_length}"
                    )
        return num_prefix_tokens

    def _hook(self, layer_idx):
        def capture(module, inputs, output):
            x = inputs[0]
            B, N, C = x.shape
            num_prefix = self._get_prefix_token_count(sequence_length=N)
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            self._kv_cache[layer_idx] = (
                qkv[1, :, :, num_prefix:].detach(),
                qkv[2, :, :, num_prefix:].detach(),
            )
        return capture

    def reset(self):
        self._kv_cache = {}

    def captured(self):
        missing = [idx for idx in self.layer_indices if idx not in self._kv_cache]
        if missing:
            raise RuntimeError(f"DINO K/V hooks missed layers {[i + 1 for i in missing]}")
        result = [self._kv_cache[idx] for idx in self.layer_indices]
        self.reset()
        return result


class EncoderKVProjection(nn.Module):
    """Project full-head DINO K/V into JiT's head layout."""
    def __init__(self, enc_dim, jit_dim, jit_heads, num_layers=1,
                 proj_type="linear", norm_type="none"):
        super().__init__()
        self.enc_dim = enc_dim
        self.jit_dim = jit_dim
        self.jit_heads = jit_heads
        self.head_dim = jit_dim // jit_heads
        if norm_type not in ("none", "layer"):
            raise ValueError("attnscaf_kv_norm must be 'none' or 'layer'")
        self.k_norms = nn.ModuleList([
            nn.LayerNorm(enc_dim) if norm_type == "layer" else nn.Identity()
            for _ in range(num_layers)
        ])
        self.v_norms = nn.ModuleList([
            nn.LayerNorm(enc_dim) if norm_type == "layer" else nn.Identity()
            for _ in range(num_layers)
        ])
        def projection():
            if proj_type == "linear":
                return nn.Linear(enc_dim, jit_dim, bias=False)
            if proj_type == "mlp":
                hidden = max(enc_dim, jit_dim)
                return nn.Sequential(nn.Linear(enc_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, jit_dim))
            raise ValueError("attnscaf_kv_proj must be 'linear' or 'mlp'")
        self.k_projs = nn.ModuleList([projection() for _ in range(num_layers)])
        self.v_projs = nn.ModuleList([projection() for _ in range(num_layers)])

    def _project(self, tensor, norm, proj, target_tokens):
        B, _, N, _ = tensor.shape
        flat = tensor.transpose(1, 2).reshape(B, N, self.enc_dim)
        if N != target_tokens:
            src, dst = int(N ** .5), int(target_tokens ** .5)
            if src * src != N or dst * dst != target_tokens:
                raise ValueError(f"Cannot resize DINO K/V from {N} to {target_tokens} tokens")
            flat = F.interpolate(
                flat.transpose(1, 2).reshape(B, self.enc_dim, src, src).float(),
                size=(dst, dst), mode="bilinear", align_corners=False,
            ).flatten(2).transpose(1, 2).to(tensor.dtype)
        out = proj(norm(flat))
        return out.reshape(B, target_tokens, self.jit_heads, self.head_dim).transpose(1, 2)

    def forward(self, kv_list, target_tokens):
        return [(
            self._project(k, self.k_norms[i], self.k_projs[i], target_tokens),
            self._project(v, self.v_norms[i], self.v_projs[i], target_tokens),
        ) for i, (k, v) in enumerate(kv_list)]
