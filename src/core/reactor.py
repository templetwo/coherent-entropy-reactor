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
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_refinement_steps = num_refinement_steps
        self.kuramoto_k = kuramoto_k
        self.target_entropy = target_entropy

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

        Approximated by variance of activations (simpler, more stable).
        """
        # Compute variance-based mass (proxy for Fisher trace)
        # Higher variance = more distributed = more "massive"
        variance = z.var(dim=-1).mean()

        # Also factor in the magnitude (energy)
        magnitude = (z ** 2).mean()

        # Combined mass metric
        mass = variance * magnitude

        return mass

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

                # Apply Kuramoto modulation
                z = self.kuramoto_step(z)

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
        target_entropy=3.0
    )

    print(f"Parameters: {count_parameters(reactor):,}")

    # Test with random distribution
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 128)
    x = F.softmax(x, dim=-1)  # Make it a distribution

    output, mass = reactor.react(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Semantic mass: {mass:.4f}")
    print(f"Mass history: {reactor.get_mass_history()}")
