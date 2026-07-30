"""
models.py
=========
Neural architectures for TWI target detection and sim-to-real adaptation.

(1) CNNBackbone  - the conv/BN/ReLU/maxpool feature extractor mirroring the
                   architecture of the baseline paper (Yadav et al., 2026, Fig. 1).
(2) BaselineCNN  - backbone + classifier head (the published baseline, retrained).
(3) GradReverse  - gradient-reversal layer for domain-adversarial training (DANN).
(4) DANN_MUSIC   - the PROPOSED model: shared backbone whose features are fused
                   with the MUSIC subspace anchor, a label classifier, and a
                   domain discriminator trained adversarially. This realises the
                   feature-level (subspace-anchored) half of the method, while the
                   physics-guided synthetic source realises the input-level half.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ----------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    """conv1-bn1-relu1-pool1 / conv2-bn2-relu2-pool2 / conv3-bn3-relu3 -> GAP."""
    def __init__(self, in_ch=1, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = width * 4

    def forward(self, x):
        return self.net(x).flatten(1)            # (B, out_dim)


# ----------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Published baseline: CNN backbone + softmax classifier (no adaptation)."""
    def __init__(self, n_classes=3, in_ch=1, width=32):
        super().__init__()
        self.backbone = CNNBackbone(in_ch, width)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, music=None):
        f = self.backbone(x)
        return self.head(f)


# ----------------------------------------------------------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ----------------------------------------------------------------------------
class DANN_MUSIC(nn.Module):
    """
    PROPOSED: Physics-guided, MUSIC-subspace-anchored domain-adversarial network.

    - backbone : shared CNN features from the range image
    - music_enc: small MLP encoding the (domain-invariant) MUSIC pseudo-spectrum
    - fused    : concat(image feature, music feature) -> shared representation
    - label_clf: target-count / material classifier (task head)
    - domain_clf: discriminates sim vs real on the GRADIENT-REVERSED fused feature,
                  forcing the representation to be domain-invariant -- but the
                  MUSIC anchor gives it a physically domain-shared component to
                  align around, stabilising adaptation.
    """
    def __init__(self, n_classes=3, music_dim=64, in_ch=1, width=32):
        super().__init__()
        self.backbone = CNNBackbone(in_ch, width)
        self.music_enc = nn.Sequential(
            nn.Linear(music_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        fused_dim = self.backbone.out_dim + 32
        self.fuse = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(0.3))
        self.label_clf = nn.Linear(128, n_classes)
        self.domain_clf = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x, music, lambd=0.0):
        fi = self.backbone(x)
        fm = self.music_enc(music)
        f = self.fuse(torch.cat([fi, fm], dim=1))
        y = self.label_clf(f)
        d = self.domain_clf(grad_reverse(f, lambd))
        return y, d, f
