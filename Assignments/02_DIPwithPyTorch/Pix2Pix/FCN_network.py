import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """Multi-scale block with optional residual shortcut."""

    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.15, use_residual=False):
        super().__init__()
        mid_channels = max(out_channels // 2, 32)
        self.use_residual = use_residual

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=drop_prob)

        if self.use_residual:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        h = self.pre(x)
        b1 = self.branch3x3(h)
        b2 = self.branch_dilated(h)
        out = self.fuse(torch.cat([b1, b2], dim=1))
        out = self.dropout(out)
        if self.use_residual:
            out = out + self.shortcut(x)
        return self.act(out)


class SemanticContextHead(nn.Module):
    """Aggregate global and multi-dilation context to improve object-level color semantics."""

    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.PReLU(channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

    def forward(self, x):
        g = self.global_pool(x)
        g = g.expand_as(x)
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), g], dim=1)
        return self.fuse(out)


class SemanticGate(nn.Module):
    """Lightweight channel-wise semantic gate."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.gate(self.avg_pool(x))
        return x * w


class SkipAttentionGate(nn.Module):
    """Attention gate for U-Net skip features conditioned on decoder features."""

    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.g_proj = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.x_proj = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.PReLU(inter_channels),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        attn = self.psi(self.g_proj(g) + self.x_proj(x))
        return x * attn


class UpCatBlock(nn.Module):
    """Upsample, concatenate with skip, then fuse by residual multi-scale block."""

    def __init__(
        self,
        in_channels,
        up_channels,
        skip_channels,
        out_channels,
        drop_prob=0.15,
        use_residual=False,
        use_attention=True,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, up_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(up_channels),
            nn.PReLU(up_channels),
        )
        if use_attention:
            self.skip_attn = SkipAttentionGate(
                g_channels=up_channels,
                x_channels=skip_channels,
                inter_channels=max(skip_channels // 2, 16),
            )
        else:
            self.skip_attn = None
        self.fuse = MultiScaleBlock(
            up_channels + skip_channels,
            out_channels,
            stride=1,
            drop_prob=drop_prob,
            use_residual=use_residual,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if self.skip_attn is not None:
            skip = self.skip_attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class FullyConvNetwork(nn.Module):
    """U-Net VAE for 256x256 RGB input/output in [-1, 1]."""

    def __init__(self, latent_channels=1024):
        super().__init__()

        # High-resolution stem for detail-preserving refinement at output stage.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        # Encoder path (U-Net downsampling): shallow blocks without residual,
        # deep blocks with residual to preserve high-level semantics.
        self.enc1 = MultiScaleBlock(32, 64, stride=2, use_residual=False)        # 256 -> 128
        self.enc2 = MultiScaleBlock(64, 128, stride=2, use_residual=False)       # 128 -> 64
        self.enc3 = MultiScaleBlock(128, 256, stride=2, use_residual=False)      # 64 -> 32
        self.enc4 = MultiScaleBlock(256, 384, stride=2, use_residual=False)      # 32 -> 16
        self.enc5 = MultiScaleBlock(384, 512, stride=2, use_residual=True)        # 16 -> 8
        self.enc6 = MultiScaleBlock(512, 768, stride=2, use_residual=True)        # 8 -> 4
        self.bottleneck = MultiScaleBlock(768, 1024, stride=1, drop_prob=0.2, use_residual=True)   # 4 -> 4
        self.semantic_context = SemanticContextHead(1024)
        self.semantic_gate_deep = SemanticGate(1024, reduction=32)
        self.bottleneck_refine = MultiScaleBlock(1024, 1024, stride=1, drop_prob=0.2, use_residual=True)

        # Latent heads.
        self.to_mu = nn.Conv2d(1024, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(1024, latent_channels, kernel_size=1)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.PReLU(1024),
        )

        # Decoder path (U-Net upsampling with skip fusion).
        self.bridge_fuse = MultiScaleBlock(1024 + 768, 1024, stride=1, drop_prob=0.2, use_residual=True)

        self.dec5 = UpCatBlock(1024, 512, 512, 512, drop_prob=0.15, use_residual=True, use_attention=True)   # 4 -> 8, skip enc5
        self.dec5_refine = MultiScaleBlock(512, 512, stride=1, drop_prob=0.15, use_residual=True)
        self.dec4 = UpCatBlock(512, 384, 384, 384, drop_prob=0.15, use_residual=True, use_attention=True)     # 8 -> 16, skip enc4
        self.dec4_refine = MultiScaleBlock(384, 384, stride=1, drop_prob=0.15, use_residual=True)
        self.dec3 = UpCatBlock(384, 256, 256, 256, drop_prob=0.12, use_residual=False, use_attention=True)    # 16 -> 32, skip enc3
        self.dec3_refine = MultiScaleBlock(256, 256, stride=1, drop_prob=0.12, use_residual=False)
        self.dec2 = UpCatBlock(256, 128, 128, 128, drop_prob=0.10, use_residual=False, use_attention=True)    # 32 -> 64, skip enc2
        self.dec1 = UpCatBlock(128, 64, 64, 64, drop_prob=0.10, use_residual=False, use_attention=True)       # 64 -> 128, skip enc1

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(48),
        )
        self.detail_fuse = MultiScaleBlock(48 + 32, 64, stride=1, drop_prob=0.06, use_residual=False)
        self.final_refine = MultiScaleBlock(64, 64, stride=1, drop_prob=0.08, use_residual=False)
        self.semantic_gate_out = SemanticGate(64, reduction=8)

        # Skip regularization helps reduce over-reliance on low-level shortcuts.
        self.skip_dropout = nn.Dropout2d(p=0.1)

        self.final_out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        b = self.bottleneck(e6)
        b = self.semantic_context(b)
        b = self.semantic_gate_deep(b)
        b = self.bottleneck_refine(b)
        mu = self.to_mu(b)
        logvar = self.to_logvar(b)
        return mu, logvar, (e0, e1, e2, e3, e4, e5, e6)

    def reparameterize(self, mu, logvar):
        # Clamp log-variance to avoid numerical overflow in exp.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        e0, e1, e2, e3, e4, e5, e6 = skips
        x = self.from_latent(z)

        # Fuse bottleneck with deepest encoder feature first (same 4x4 scale).
        if self.training:
            e6 = self.skip_dropout(e6)
        x = self.bridge_fuse(torch.cat([x, e6], dim=1))

        if self.training:
            e5 = self.skip_dropout(e5)
            e4 = self.skip_dropout(e4)
            e3 = self.skip_dropout(e3)
            e2 = self.skip_dropout(e2)
            e1 = self.skip_dropout(e1)

        x = self.dec5(x, e5)
        x = self.dec5_refine(x)
        x = self.dec4(x, e4)
        x = self.dec4_refine(x)
        x = self.dec3(x, e3)
        x = self.dec3_refine(x)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        x = self.final_up(x)
        x = self.detail_fuse(torch.cat([x, e0], dim=1))
        x = self.final_refine(x)
        x = self.semantic_gate_out(x)
        return self.final_out(x)

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            # Deterministic latent during eval gives stable validation metrics.
            z = mu
        recon = self.decode(z, skips)
        return recon, mu, logvar
    