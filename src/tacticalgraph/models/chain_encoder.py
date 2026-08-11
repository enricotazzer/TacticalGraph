"""Module 4's learned sequence representation: a GRU autoencoder over possession chains.

Self-supervised on purpose. There is no ground-truth label for "style of play", so the
encoder is trained to reconstruct the action sequence and the latent space is then clustered.
Using `ends_in_shot` as a training signal would collapse the task into shot prediction and
make the later "which patterns precede shots?" analysis circular.

A GRU rather than a Transformer: chains are short (median 3 actions, p90 ~9-10), so
recurrence is both cheaper and better matched than attention, and there is far less to
overfit. Trained on training-split chains only, then applied to all.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

LATENT_DIM = 32


class ChainAutoencoder(nn.Module):
    """Encode a padded action sequence to a fixed vector, then reconstruct it.

    The decoder is conditioned only on the latent vector (it re-reads the latent at every
    step rather than being fed the true previous token), so the latent has to carry the whole
    chain. Teacher forcing would let the decoder cheat and leave the latent underused.
    """

    def __init__(
        self,
        n_token_features: int,
        hidden_dim: int = 64,
        latent_dim: int = LATENT_DIM,
        max_length: int = 12,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(n_token_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.reconstruct = nn.Linear(hidden_dim, n_token_features)
        self.max_length = max_length
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        return self.to_latent(hidden.squeeze(0))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        repeated = latent.unsqueeze(1).repeat(1, self.max_length, 1)
        hidden = torch.tanh(self.from_latent(latent)).unsqueeze(0).contiguous()
        output, _ = self.decoder(repeated, hidden)
        return self.reconstruct(output)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, lengths)
        return self.decode(latent), latent


def _mask_for(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """(batch, max_length, 1) mask so padding contributes nothing to the loss."""
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return (steps < lengths.unsqueeze(1)).float().unsqueeze(-1)


def train_chain_encoder(
    sequences: np.ndarray,
    lengths: np.ndarray,
    train_mask: np.ndarray,
    latent_dim: int = LATENT_DIM,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 2e-3,
    val_fraction: float = 0.1,
    patience: int = 6,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[ChainAutoencoder, dict[str, list[float]]]:
    """Fit the autoencoder on training-split chains only.

    `val_fraction` carves a small validation slice out of the *training* chains purely for
    early stopping. That is safe here in a way it would not be for Module 3: the objective is
    reconstruction, not the reported target, so a chain-level split leaks nothing about
    `ends_in_shot` or about the held-out season.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_indices = np.flatnonzero(train_mask)
    rng.shuffle(train_indices)
    cut = max(int(len(train_indices) * (1 - val_fraction)), 1)
    fit_indices, val_indices = train_indices[:cut], train_indices[cut:]

    x_all = torch.as_tensor(sequences, dtype=torch.float32)
    length_all = torch.as_tensor(lengths, dtype=torch.long)
    max_length = sequences.shape[1]

    model = ChainAutoencoder(
        n_token_features=sequences.shape[2],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_length=max_length,
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    def epoch_loss(indices: np.ndarray, train: bool) -> float:
        model.train(train)
        total, seen = 0.0, 0
        order = rng.permutation(indices) if train else indices
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = x_all[batch].to(device)
            n = length_all[batch].to(device)
            mask = _mask_for(n, max_length)
            with torch.set_grad_enabled(train):
                reconstruction, _ = model(x, n)
                loss = ((reconstruction - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += float(loss) * len(batch)
            seen += len(batch)
        return total / max(seen, 1)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state, best_loss, stale = None, float("inf"), 0

    for epoch in range(epochs):
        train_loss = epoch_loss(fit_indices, train=True)
        val_loss = epoch_loss(val_indices, train=False) if len(val_indices) else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log.info("chain encoder early stop at epoch %d (val MSE %.5f)", epoch, best_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(
        "chain encoder trained on %d chains (%d held out for early stopping); val MSE %.5f",
        len(fit_indices), len(val_indices), best_loss,
    )
    return model, history


@torch.no_grad()
def encode_all(
    model: ChainAutoencoder,
    sequences: np.ndarray,
    lengths: np.ndarray,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """Latent vector for every chain, in input order."""
    model.eval()
    x_all = torch.as_tensor(sequences, dtype=torch.float32)
    length_all = torch.as_tensor(lengths, dtype=torch.long)
    chunks = []
    for start in range(0, len(sequences), batch_size):
        stop = start + batch_size
        latent = model.encode(x_all[start:stop].to(device), length_all[start:stop].to(device))
        chunks.append(latent.cpu().numpy())
    return np.concatenate(chunks, axis=0)
