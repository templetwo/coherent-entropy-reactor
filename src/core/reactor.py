"""
Coherent Entropy Reactor - Core Architecture

A recursive small network with liquid dynamics that:
- Measures its own semantic mass
- Resists perturbation via adaptive loops
- Emerges coherence from chaos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RecursiveRefinementBlock(nn.Module):
    """
    TRM-style recursive refinement block.

    Refines latent state z through multiple passes,
    accumulating semantic mass at each step.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention for internal coherence
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward for state transformation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Refine latent state z.

        Args:
            z: Latent state [batch, seq, hidden]

        Returns:
            Refined state z'
        """
        # Self-attention with residual
        z_norm = self.norm1(z)
        attn_out, _ = self.self_attn(z_norm, z_norm, z_norm)
        z = z + attn_out

        # FFN with residual
        z = z + self.ffn(self.norm2(z))

        return z


class EntropyEncoder(nn.Module):
    """
    Encodes input probability distributions into latent entropy field.

    Unlike token embeddings, this operates on continuous distributions.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Project distribution parameters to hidden space
        self.proj = nn.Linear(input_dim, hidden_dim)

        # Entropy-aware normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Encode probability distribution.

        Args:
            dist: Input distribution [batch, seq, input_dim]

        Returns:
            Entropy field z [batch, seq, hidden]
        """
        z = self.proj(dist)
        z = self.norm(z)
        return z


class CoherentEntropyReactor(nn.Module):
    """
    The Coherent Entropy Reactor.

    A recursive network that:
    1. Encodes entropy distributions (not tokens)
    2. Refines through liquid dynamics
    3. Measures its own semantic mass
    4. Outputs reactions + evolved state

    Args:
        input_dim: Dimension of input distributions
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        num_layers: Number of recursive refinement layers
        num_refinement_steps: Recursive passes per layer
        kuramoto_k: Kuramoto coupling strength
        target_entropy: Target entropy in nats (2-4)
        drift_strength: Noise scale for cage escape (0.1 default)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_refinement_steps: int = 3,
        kuramoto_k: float = 2.0,
        target_entropy: float = 3.0,
        drift_strength: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_refinement_steps = num_refinement_steps
        self.kuramoto_k = kuramoto_k
        self.target_entropy = target_entropy
        self.drift_strength = drift_strength

        # Entropy encoder
        self.encoder = EntropyEncoder(input_dim, hidden_dim)

        # Recursive refinement layers
        self.layers = nn.ModuleList([
            RecursiveRefinementBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Semantic mass tracking
        self.mass_history = []

        # Phase state for Kuramoto coupling
        self.register_buffer('phase', torch.zeros(1))

    def compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy of latent state distribution."""
        # Treat z as logits, compute softmax entropy
        probs = F.softmax(z, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()

    def compute_semantic_mass(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic mass via Fisher Information trace.

        M_semantic = (1/N) * Tr(I(θ))

        Fisher Information I = E[∇log p · ∇log p^T]
        For softmax distribution, this measures curvature in probability space.
        """
        # Ensure z requires grad for Fisher computation
        z_fisher = z.detach().requires_grad_(True)

        # Compute log probabilities
        log_probs = F.log_softmax(z_fisher, dim=-1)

        # Fisher = E[||∇log p||²] - trace of Fisher Information matrix
        # Sum over all dimensions to get scalar for gradient
        log_prob_sum = log_probs.sum()

        # Compute gradient
        grad = torch.autograd.grad(
            log_prob_sum,
            z_fisher,
            create_graph=False,
            retain_graph=False
        )[0]

        # Fisher trace approximation: mean of squared gradients
        # This is Tr(I(θ)) / N
        fisher_trace = (grad ** 2).mean()

        return fisher_trace

    def apply_drift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Escape entropy cage via latent perturbation.

        When entropy falls below target, inject noise proportional
        to the deficit. This is the corrective mechanism alongside
        Kuramoto's oscillatory modulation.
        """
        with torch.no_grad():
            current_entropy = self.compute_entropy(z)

            if current_entropy < self.target_entropy:
                # Deficit-proportional noise injection
                deficit = self.target_entropy - current_entropy
                noise = torch.randn_like(z) * deficit * self.drift_strength
                z = z + noise

        return z

    def kuramoto_step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply Kuramoto oscillator dynamics to modulate state.

        T = T_base + A * sin(φ_mean)
        """
        # Update phase based on current entropy
        current_entropy = self.compute_entropy(z)
        phase_delta = self.kuramoto_k * torch.sin(self.target_entropy - current_entropy)
        self.phase = self.phase + phase_delta * 0.1

        # Modulate z based on phase
        modulation = 1.0 + 0.3 * torch.sin(self.phase)
        return z * modulation

    def forward(
        self,
        x: torch.Tensor,
        return_mass: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        React to input distribution.

        Args:
            x: Input distribution [batch, seq, input_dim]
            return_mass: Whether to compute and return semantic mass

        Returns:
            y: Output reaction [batch, seq, output_dim]
            mass: Semantic mass (if return_mass=True)
        """
        # Encode input to entropy field
        z = self.encoder(x)

        # Track mass through refinement
        masses = []

        # Recursive refinement with liquid dynamics
        for layer in self.layers:
            for _ in range(self.num_refinement_steps):
                # Refine state
                z = layer(z)

                # Apply Kuramoto modulation (oscillatory)
                z = self.kuramoto_step(z)

                # Apply drift if trapped in cage (corrective)
                z = self.apply_drift(z)

                # Track mass if requested
                if return_mass:
                    mass = self.compute_semantic_mass(z)
                    masses.append(mass)

        # Project to output
        y = self.output_proj(z)

        # Compute final mass
        final_mass = None
        if return_mass and masses:
            final_mass = torch.stack(masses).mean()
            self.mass_history.append(final_mass.item())

        return y, final_mass

    def react(
        self,
        input_dist: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Simplified interface for reaction.

        Args:
            input_dist: Input probability distribution

        Returns:
            output: Reaction output
            mass: Current semantic mass
        """
        output, mass = self.forward(input_dist, return_mass=True)
        return output, mass.item() if mass is not None else 0.0

    def get_mass_history(self):
        """Return history of semantic mass measurements."""
        return self.mass_history

    def reset_mass_history(self):
        """Clear mass history."""
        self.mass_history = []


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Coherent Entropy Reactor - Test")
    print("=" * 50)

    reactor = CoherentEntropyReactor(
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
        num_refinement_steps=3,
        kuramoto_k=2.0,
        target_entropy=3.0,
        drift_strength=0.1
    )

    print(f"Parameters: {count_parameters(reactor):,}")

    # Test with random distribution
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 128)
    x = F.softmax(x, dim=-1)  # Make it a distribution

    output, mass = reactor.react(x)

    # Compute output entropy
    probs = F.softmax(output, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Semantic mass (Fisher): {mass:.6f}")
    print(f"Output entropy: {entropy:.2f} nats")
    print(f"Phase state: {reactor.phase.item():.4f} rad")
    print(f"Mass history length: {len(reactor.get_mass_history())}")
