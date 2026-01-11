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
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import math


@dataclass
class RefinementStep:
    """Metrics captured at each refinement step."""
    layer_idx: int
    step_idx: int
    entropy: float
    mass: float
    phase: float
    drift_applied: bool = False
    deficit: float = 0.0


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
        max_drift_deficit: Maximum deficit for drift clamping (2.0 default)
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
        max_drift_deficit: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_refinement_steps = num_refinement_steps
        self.kuramoto_k = kuramoto_k
        self.target_entropy = target_entropy
        self.drift_strength = drift_strength
        self.max_drift_deficit = max_drift_deficit

        # Entropy encoder
        self.encoder = EntropyEncoder(input_dim, hidden_dim)

        # Recursive refinement layers
        self.layers = nn.ModuleList([
            RecursiveRefinementBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Trajectory tracking (per-step metrics)
        self.trajectory: List[RefinementStep] = []

        # Legacy mass history (for backward compat)
        self.mass_history: List[float] = []

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
        Compute semantic mass via analytic Fisher Information proxy.

        For categorical distribution p = softmax(z), the Fisher trace proxy is:
            1 - sum_i(p_i^2)

        Properties:
        - Near 0 when distribution is peaked (low entropy, low mass)
        - Near 1 - 1/K when uniform (high entropy, high mass)
        - No autograd needed, fast and stable
        - Theoretically aligned with Fisher Information for softmax
        """
        probs = F.softmax(z, dim=-1)
        # Fisher proxy: 1 - sum(p^2) per position, averaged
        fisher_proxy = (1.0 - (probs ** 2).sum(dim=-1)).mean()
        return fisher_proxy

    def apply_drift(self, z: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """
        Escape entropy cage via latent perturbation.

        When entropy falls below target, inject noise proportional
        to the deficit (clamped to prevent runaway).

        Returns:
            z: Modified tensor
            drift_applied: Whether drift was applied
            deficit: The entropy deficit (clamped)
        """
        with torch.no_grad():
            current_entropy = self.compute_entropy(z)

            if current_entropy < self.target_entropy:
                # Compute and clamp deficit to prevent runaway noise
                raw_deficit = self.target_entropy - current_entropy
                deficit = min(raw_deficit.item(), self.max_drift_deficit)

                # Deficit-proportional noise injection
                noise = torch.randn_like(z) * deficit * self.drift_strength
                z = z + noise
                return z, True, deficit

        return z, False, 0.0

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
        return_mass: bool = True,
        track_trajectory: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        React to input distribution.

        Args:
            x: Input distribution [batch, seq, input_dim]
            return_mass: Whether to compute and return semantic mass
            track_trajectory: Whether to log per-step metrics

        Returns:
            y: Output reaction [batch, seq, output_dim]
            mass: Semantic mass (if return_mass=True)
        """
        # Encode input to entropy field
        z = self.encoder(x)

        # Track mass through refinement
        masses = []

        # Recursive refinement with liquid dynamics
        for layer_idx, layer in enumerate(self.layers):
            for step_idx in range(self.num_refinement_steps):
                # Refine state
                z = layer(z)

                # Apply Kuramoto modulation (oscillatory)
                z = self.kuramoto_step(z)

                # Apply drift if trapped in cage (corrective)
                z, drift_applied, deficit = self.apply_drift(z)

                # Compute metrics
                if return_mass or track_trajectory:
                    mass = self.compute_semantic_mass(z)
                    entropy = self.compute_entropy(z).item()
                    masses.append(mass)

                    # Log trajectory step
                    if track_trajectory:
                        self.trajectory.append(RefinementStep(
                            layer_idx=layer_idx,
                            step_idx=step_idx,
                            entropy=entropy,
                            mass=mass.item(),
                            phase=self.phase.item(),
                            drift_applied=drift_applied,
                            deficit=deficit
                        ))

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

    def get_mass_history(self) -> List[float]:
        """Return history of semantic mass measurements."""
        return self.mass_history

    def get_trajectory(self) -> List[RefinementStep]:
        """Return full trajectory of refinement steps."""
        return self.trajectory

    def reset_history(self):
        """Clear all history (mass and trajectory)."""
        self.mass_history = []
        self.trajectory = []

    def reset_mass_history(self):
        """Clear mass history (legacy compat)."""
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
        drift_strength=0.1,
        max_drift_deficit=2.0
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

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Semantic mass (Fisher proxy): {mass:.6f}")
    print(f"Output entropy: {entropy:.2f} nats")
    print(f"Phase state: {reactor.phase.item():.4f} rad")

    # Show trajectory
    trajectory = reactor.get_trajectory()
    print(f"\nTrajectory ({len(trajectory)} steps):")
    print(f"{'Layer':<6} {'Step':<5} {'Entropy':<10} {'Mass':<10} {'Phase':<10} {'Drift':<6}")
    print("-" * 55)
    for step in trajectory:
        drift_str = f"{step.deficit:.2f}" if step.drift_applied else "-"
        print(f"{step.layer_idx:<6} {step.step_idx:<5} {step.entropy:<10.3f} {step.mass:<10.4f} {step.phase:<10.4f} {drift_str:<6}")
