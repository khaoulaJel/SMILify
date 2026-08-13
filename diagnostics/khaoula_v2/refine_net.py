"""Shared RefineNet architecture, imported by both the training script and the eval
wiring in within_part_signal_dino.py -- one definition, so train and eval can never
silently drift apart.

Small point-wise autoencoder (Uzolas et al. SIGGRAPH Asia 2025, arXiv 2503.18254):
encoder 768->256->128, decoder 128->256->768. Trained with reconstruction + a
geodesic-distance-guided contrastive term (see train_refine_autoencoder.py) so that
vertex pairs far apart on the mesh surface are pushed apart in embedding space even
when their base DINO features are near-identical.
"""

import torch
import torch.nn as nn


class RefineNet(nn.Module):
    def __init__(self, in_dim=768, embed_dim=128, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
