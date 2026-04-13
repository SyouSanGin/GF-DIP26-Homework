import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SemanticInject(nn.Module):
    """Inject semantic-map guidance into decoder features at matched resolution."""

    def __init__(self, semantic_channels, feat_channels):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.PReLU(feat_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.PReLU(feat_channels),
        )

    def forward(self, feat, semantic_map):
        sem = F.interpolate(semantic_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        sem = self.semantic_proj(sem)
        g = self.gate(torch.cat([feat, sem], dim=1))
        fused = torch.cat([feat, sem * g], dim=1)
        return self.fuse(fused)


class EncoderDownBlock(nn.Module):
    """Downsampling block for the semantic encoder."""

    def __init__(self, in_channels, out_channels, drop_prob=0.12, use_residual=False):
        super().__init__()
        self.block = MultiScaleBlock(
            in_channels,
            out_channels,
            stride=2,
            drop_prob=drop_prob,
            use_residual=use_residual,
        )

    def forward(self, x):
        return self.block(x)


class DecoderUpBlock(nn.Module):
    """Upsample and refine without any U-Net skip connections."""

    def __init__(self, in_channels, out_channels, drop_prob=0.12, use_residual=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.refine = MultiScaleBlock(
            out_channels,
            out_channels,
            stride=1,
            drop_prob=drop_prob,
            use_residual=use_residual,
        )

    def forward(self, x):
        x = self.up(x)
        return self.refine(x)


class SkipFuse(nn.Module):
    """Fuse encoder detail features into decoder features with a learned gate."""

    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.enc_proj = nn.Sequential(
            nn.Conv2d(enc_channels, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.PReLU(dec_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.PReLU(dec_channels),
        )

    def forward(self, dec_feat, enc_feat):
        enc = self.enc_proj(enc_feat)
        g = self.gate(torch.cat([dec_feat, enc], dim=1))
        return self.fuse(torch.cat([dec_feat, enc * g], dim=1))


class FullyConvNetwork(nn.Module):
    """Conditional VAE for 256x256 edge-map to RGB image synthesis."""

    def __init__(self, latent_channels=256):
        super().__init__()

        # Edge encoder: encode edge structure into a compact latent.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        self.enc1 = EncoderDownBlock(32, 64, drop_prob=0.08, use_residual=False)      # 256 -> 128
        self.enc2 = EncoderDownBlock(64, 128, drop_prob=0.08, use_residual=False)     # 128 -> 64
        self.enc3 = EncoderDownBlock(128, 256, drop_prob=0.10, use_residual=False)    # 64 -> 32
        self.enc4 = EncoderDownBlock(256, 384, drop_prob=0.10, use_residual=True)     # 32 -> 16
        self.enc5 = EncoderDownBlock(384, 512, drop_prob=0.12, use_residual=True)     # 16 -> 8
        self.enc6 = EncoderDownBlock(512, 768, drop_prob=0.12, use_residual=True)     # 8 -> 4
        self.bottleneck = MultiScaleBlock(768, 1024, stride=1, drop_prob=0.2, use_residual=True)
        self.semantic_context = SemanticContextHead(1024)
        self.semantic_gate_deep = SemanticGate(1024, reduction=32)
        self.bottleneck_refine = MultiScaleBlock(1024, 1024, stride=1, drop_prob=0.15, use_residual=True)

        # VAE latent heads.
        self.to_mu = nn.Conv2d(1024, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(1024, latent_channels, kernel_size=1)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.PReLU(1024),
        )

        # Decoder path.
        self.dec6 = DecoderUpBlock(1024, 768, drop_prob=0.16, use_residual=True)     # 4 -> 8
        self.dec5 = DecoderUpBlock(768, 512, drop_prob=0.15, use_residual=True)      # 8 -> 16
        self.dec4 = DecoderUpBlock(512, 384, drop_prob=0.14, use_residual=True)      # 16 -> 32
        self.dec3 = DecoderUpBlock(384, 256, drop_prob=0.12, use_residual=True)      # 32 -> 64
        self.dec2 = DecoderUpBlock(256, 128, drop_prob=0.10, use_residual=False)     # 64 -> 128
        self.dec1 = DecoderUpBlock(128, 64, drop_prob=0.08, use_residual=False)      # 128 -> 256

        # Re-introduce shallow detail paths for edge-to-RGB reconstruction.
        self.skip_fuse_64 = SkipFuse(enc_channels=128, dec_channels=256)
        self.skip_fuse_128 = SkipFuse(enc_channels=64, dec_channels=128)
        self.skip_fuse_256 = SkipFuse(enc_channels=32, dec_channels=64)

        # Inject semantic constraints at multiple decoder scales.
        self.semantic_inject_8 = SemanticInject(semantic_channels=3, feat_channels=768)
        self.semantic_inject_16 = SemanticInject(semantic_channels=3, feat_channels=512)
        self.semantic_inject_32 = SemanticInject(semantic_channels=3, feat_channels=384)
        self.semantic_inject_64_pre = SemanticInject(semantic_channels=3, feat_channels=256)
        self.semantic_inject_64 = SemanticInject(semantic_channels=3, feat_channels=128)
        self.semantic_inject_128 = SemanticInject(semantic_channels=3, feat_channels=64)
        self.semantic_inject_256 = SemanticInject(semantic_channels=3, feat_channels=64)

        # Auxiliary predictions provide deep supervision for sharper outputs.
        self.aux_head_64 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.aux_head_128 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.output_refine = nn.Sequential(
            MultiScaleBlock(64, 64, stride=1, drop_prob=0.06, use_residual=False),
            SemanticGate(64, reduction=8),
            MultiScaleBlock(64, 64, stride=1, drop_prob=0.06, use_residual=False),
        )

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
        skips = {
            "e0": e0,
            "e1": e1,
            "e2": e2,
        }
        return mu, logvar, skips

    def reparameterize(self, mu, logvar, noise_scale=1.0):
        # Clamp log-variance to avoid numerical overflow in exp.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-4, max=1.0)
        eps = torch.randn_like(std)
        return mu + eps * std * noise_scale

    def decode(self, z, edge_input, skips):
        x = self.from_latent(z)

        x = self.dec6(x)
        x = self.semantic_inject_8(x, edge_input)
        x = self.dec5(x)
        x = self.semantic_inject_16(x, edge_input)
        x = self.dec4(x)
        x = self.semantic_inject_32(x, edge_input)
        x = self.dec3(x)
        x = self.semantic_inject_64_pre(x, edge_input)
        x = self.skip_fuse_64(x, skips["e2"])
        x = self.dec2(x)
        x = self.semantic_inject_64(x, edge_input)
        x = self.skip_fuse_128(x, skips["e1"])
        aux_64 = torch.tanh(
            F.interpolate(self.aux_head_64(x), size=edge_input.shape[-2:], mode="bilinear", align_corners=False)
        )
        x = self.dec1(x)
        x = self.semantic_inject_128(x, edge_input)
        x = self.skip_fuse_256(x, skips["e0"])
        aux_128 = torch.tanh(
            F.interpolate(self.aux_head_128(x), size=edge_input.shape[-2:], mode="bilinear", align_corners=False)
        )
        x = self.output_refine(x)
        x = self.semantic_inject_256(x, edge_input)
        return self.final_out(x), [aux_128, aux_64]

    def forward(self, x, latent_noise_scale=1.0):
        mu, logvar, skips = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar, noise_scale=latent_noise_scale)
        else:
            # Deterministic latent during eval gives stable validation metrics.
            z = mu
        recon, aux_outputs = self.decode(z, x, skips)
        return recon, mu, logvar, aux_outputs
    