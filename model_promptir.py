from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=1, keepdim=True, unbiased=False)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class GatedDWFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.pw1 = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2
        )
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.pw2(x)


class MDTA(nn.Module):
    """A compact multi-DConv head transposed self-attention variant for restoration."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, c, h, w)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDWFFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PromptGenerator(nn.Module):
    """
    Generates content-adaptive prompts from global features and injects them into decoder features.
    This keeps one model specialized for multiple degradations (rain/snow) without separate networks.
    """

    def __init__(self, dim: int, prompt_dim: int = 64, num_prompts: int = 8) -> None:
        super().__init__()
        self.prompt_bank = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, num_prompts, kernel_size=1),
        )
        self.project = nn.Sequential(
            nn.Conv2d(prompt_dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = self.router(x).flatten(1)
        weights = logits.softmax(dim=-1)
        prompt_vec = torch.matmul(weights, self.prompt_bank)
        prompt = prompt_vec[:, :, None, None].expand(b, -1, h, w)
        return self.project(prompt)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PromptIRNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 48,
        blocks=(2, 3, 4, 3),
        heads=(1, 2, 4, 2),
        prompt_dim: int = 64,
        num_prompts: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)

        self.enc1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0]) for _ in range(blocks[0])]
        )
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1]) for _ in range(blocks[1])]
        )
        self.down2 = Downsample(dim * 2, dim * 4)

        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2]) for _ in range(blocks[2])]
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.fuse1 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.dec1 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[3]) for _ in range(blocks[3])]
        )

        self.up2 = Upsample(dim * 2, dim)
        self.fuse2 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.dec2 = nn.Sequential(
            *[TransformerBlock(dim, heads[0]) for _ in range(blocks[0])]
        )

        self.prompt1 = PromptGenerator(
            dim * 2, prompt_dim=prompt_dim, num_prompts=num_prompts
        )
        self.prompt2 = PromptGenerator(
            dim, prompt_dim=prompt_dim, num_prompts=num_prompts
        )

        self.out = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.embed(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        z = self.latent(self.down2(e2))

        d1 = self.up1(z)
        d1 = self.fuse1(torch.cat([d1, e2], dim=1))
        d1 = d1 + self.prompt1(d1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self.fuse2(torch.cat([d2, e1], dim=1))
        d2 = d2 + self.prompt2(d2)
        d2 = self.dec2(d2)

        out = self.out(d2) + inp
        return out.clamp(0.0, 1.0)
