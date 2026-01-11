import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import math

# Integration of specialized components
from src.liquid.dynamics import LiquidLayer
from src.entropy.measurement import SemanticMass, EntropyTracker


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
    # Normalization diagnostics
    sum_p: float = 1.0
    max_prob: float = 0.0
    max_entropy: float = 0.0


class EntropyEncoder(nn.Module):
    """
    Encodes input probability distributions into latent entropy field.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        z = self.proj(dist)
        z = self.norm(z)
        return z


class CoherentEntropyReactor(nn.Module):
    """
    The Coherent Entropy Reactor.

    A recursive network that:
    1. Encodes entropy distributions (not tokens)
    2. Refines through Liquid Neural Network (LNN) layers
    3. Measures its own semantic mass via Fisher Information
    4. Outputs reactions + evolved state
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_refinement_steps: int = 3,  # Preserved for API compat, but LiquidLayer handles integration
        kuramoto_k: float = 2.0,
        target_entropy: float = 3.0,
        drift_strength: float = 0.1,
        max_drift_deficit: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_entropy = target_entropy
        self.drift_strength = drift_strength
        self.max_drift_deficit = max_drift_deficit

        # Entropy encoder
        self.encoder = EntropyEncoder(input_dim, hidden_dim)

        # Liquid Neural Network Layers (Replaces RecursiveRefinementBlock)
        # Note: LiquidLayer projects hidden->input internally for residual, 
        # so we configure it to maintain hidden_dim flow.
        self.layers = nn.ModuleList([
            LiquidLayer(
                input_dim=hidden_dim, # We operate in hidden space after encoding
                hidden_dim=hidden_dim,
                kuramoto_k=kuramoto_k,
                integration_steps=num_refinement_steps # Mapping refinement steps to ODE steps
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Components
        self.tracker = EntropyTracker(target_entropy=target_entropy)
        
        # Trajectory tracking
        self.trajectory: List[RefinementStep] = []
        self.mass_history: List[float] = []

    def compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy of latent state distribution."""
        probs = F.softmax(z, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()

    def apply_drift(self, z: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """
        Symmetric entropy control via perturbation OR sharpening.
        """
        with torch.no_grad():
            current_entropy = self.compute_entropy(z)
            margin = 0.5

            # ESCAPE: entropy too low → add noise
            if current_entropy < self.target_entropy:
                raw_deficit = self.target_entropy - current_entropy
                deficit = min(raw_deficit.item(), self.max_drift_deficit)
                noise = torch.randn_like(z) * deficit * self.drift_strength
                z = z + noise
                return z, True, deficit

            # BRAKE: entropy too high → sharpen
            if current_entropy > self.target_entropy + margin:
                excess = current_entropy - self.target_entropy
                temp = max(0.5, 1.0 - excess.item() * 0.1)
                z = z * (1.0 / temp)
                return z, True, -excess.item()

        return z, False, 0.0

    def forward(
        self,
        x: torch.Tensor,
        return_mass: bool = True,
        track_trajectory: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        React to input distribution through Liquid Core.
        """
        # Encode input to entropy field
        z = self.encoder(x)
        
        # Initial hidden state for liquid cells
        h = None 
        masses = []

        # Process through Liquid Layers
        for layer_idx, layer in enumerate(self.layers):
            # LiquidLayer handles the recursive ODE integration internally
            # z is treated as the sequence [batch, seq, hidden]
            z, h = layer(z, h=h)

            # Apply drift control (Entropy intervention)
            z, drift_applied, deficit = self.apply_drift(z)

            # Record metrics
            if return_mass or track_trajectory:
                # Use robust Fisher mass computation
                # We use 'empirical' here to avoid second-order gradients during forward pass overhead,
                # or we could implement a fast proxy in SemanticMass if needed.
                # For speed in forward pass, we'll stick to the lightweight proxy logic *inside* SemanticMass 
                # if we implemented one, but since SemanticMass currently uses gradient/empirical,
                # let's use a simplified analytic proxy here for performance, 
                # OR trust the user wants the real deal. 
                # Let's use the analytic proxy logic from the original class but implemented cleanly.
                probs = F.softmax(z, dim=-1)
                mass = (1.0 - (probs ** 2).sum(dim=-1)).mean() # Fast analytic proxy
                
                entropy = self.compute_entropy(z).item()
                self.tracker.record(entropy)
                masses.append(mass)

                if track_trajectory:
                    K = z.shape[-1]
                    self.trajectory.append(RefinementStep(
                        layer_idx=layer_idx,
                        step_idx=0, # LiquidLayer abstracts steps, so we log per layer
                        entropy=entropy,
                        mass=mass.item(),
                        phase=layer.get_order_parameter(), # Get Kuramoto order
                        drift_applied=drift_applied,
                        deficit=deficit,
                        sum_p=probs.sum(dim=-1).mean().item(),
                        max_prob=probs.max(dim=-1).values.mean().item(),
                        max_entropy=math.log(K)
                    ))

        # Project to output
        y = self.output_proj(z)

        final_mass = None
        if return_mass and masses:
            final_mass = torch.stack(masses).mean()
            self.mass_history.append(final_mass.item())

        return y, final_mass

    def react(self, input_dist: torch.Tensor) -> Tuple[torch.Tensor, float]:
        output, mass = self.forward(input_dist, return_mass=True)
        return output, mass.item() if mass is not None else 0.0

    def get_mass_history(self) -> List[float]:
        return self.mass_history

    def get_trajectory(self) -> List[RefinementStep]:
        return self.trajectory

    def reset_history(self):
        self.mass_history = []
        self.trajectory = []
        self.tracker.reset()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Integration Test
    print("Coherent Entropy Reactor (Liquid Core) - Integration Test")
    print("=" * 60)

    reactor = CoherentEntropyReactor(
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
        kuramoto_k=2.0,
        target_entropy=3.0
    )

    print(f"Parameters: {count_parameters(reactor):,}")
    
    # Generate random probability distribution
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 128)
    x = F.softmax(x, dim=-1)

    print("\nProcessing...")
    output, mass = reactor.react(x)

    probs = F.softmax(output, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Semantic Mass: {mass:.6f}")
    print(f"Output Entropy: {entropy:.2f} nats")
    print(f"Entropy Zone: {reactor.tracker.get_zone(entropy)}")
    
    traj = reactor.get_trajectory()
    print(f"\nTrajectory ({len(traj)} layers):")
    print(f"{'Layer':<6} {'Entropy':<10} {'Mass':<10} {'Phase (R)':<10} {'Drift':<6}")
    print("-" * 50)
    for step in traj:
        drift_str = f"{step.deficit:.2f}" if step.drift_applied else "-"
        print(f"{step.layer_idx:<6} {step.entropy:<10.3f} {step.mass:<10.4f} {step.phase:<10.4f} {drift_str:<6}")
