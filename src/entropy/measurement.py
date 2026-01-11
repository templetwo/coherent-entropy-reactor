"""
Semantic Mass and Entropy Measurement

Implements MCC formulas:
- Fisher Information Mass
- Commutation Cost
- Entropy tracking
"""

import torch
import torch.nn.functional as F
from typing import Callable, List, Optional
import numpy as np


class EntropyTracker:
    """Tracks entropy over time with zone classification."""

    # Entropy zones (in nats)
    ZONES = {
        'LASER': (0.0, 2.0),      # Over-constrained
        'CAGE': (2.0, 3.5),       # RLHF artifact zone
        'LANTERN': (3.5, 5.0),    # Target zone - creative
        'CHAOS': (5.0, float('inf'))  # Uncontrolled
    }

    def __init__(self, target_entropy: float = 3.5):
        self.target = target_entropy
        self.history: List[float] = []

    def record(self, entropy: float):
        """Record an entropy measurement."""
        self.history.append(entropy)

    def get_zone(self, entropy: float) -> str:
        """Classify entropy into zone."""
        for zone, (low, high) in self.ZONES.items():
            if low <= entropy < high:
                return zone
        return 'UNKNOWN'

    def zone_residence(self) -> dict:
        """Calculate time spent in each zone."""
        if not self.history:
            return {}

        counts = {zone: 0 for zone in self.ZONES}
        for e in self.history:
            zone = self.get_zone(e)
            if zone in counts:
                counts[zone] += 1

        total = len(self.history)
        return {zone: count / total for zone, count in counts.items()}

    def mean(self) -> float:
        """Mean entropy."""
        return np.mean(self.history) if self.history else 0.0

    def std(self) -> float:
        """Entropy standard deviation."""
        return np.std(self.history) if len(self.history) > 1 else 0.0

    def reset(self):
        """Clear history."""
        self.history = []


class SemanticMass:
    """
    Computes semantic mass via Fisher Information.

    M_semantic = (1/N) * Tr(I(θ))

    Where I(θ) is the Fisher Information Matrix.
    """

    @staticmethod
    def compute(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        method: str = 'gradient'
    ) -> float:
        """
        Compute semantic mass of model given inputs.

        Args:
            model: The model to measure
            inputs: Input tensor
            method: 'gradient' or 'empirical'

        Returns:
            Semantic mass (scalar)
        """
        if method == 'gradient':
            return SemanticMass._gradient_method(model, inputs)
        elif method == 'empirical':
            return SemanticMass._empirical_method(model, inputs)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _gradient_method(model: torch.nn.Module, inputs: torch.Tensor) -> float:
        """
        Compute mass via gradient magnitudes.

        Approximates Fisher trace as sum of squared gradients.
        """
        model.eval()

        # Forward pass
        inputs = inputs.requires_grad_(True)
        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Compute log-prob (treat outputs as logits)
        log_probs = F.log_softmax(outputs, dim=-1)

        # Sum for scalar loss
        loss = log_probs.sum()

        # Compute gradients
        loss.backward()

        # Fisher trace = sum of squared gradients over all parameters
        fisher_trace = 0.0
        n_params = 0

        for param in model.parameters():
            if param.grad is not None:
                fisher_trace += (param.grad ** 2).sum().item()
                n_params += param.numel()

        # Normalize
        mass = fisher_trace / max(n_params, 1)

        # Clear gradients
        model.zero_grad()

        return mass

    @staticmethod
    def _empirical_method(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        n_samples: int = 10
    ) -> float:
        """
        Compute mass via empirical Fisher.

        Sample from model outputs and compute variance.
        """
        model.eval()
        masses = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Add small perturbation
                perturbed = inputs + torch.randn_like(inputs) * 0.1
                outputs = model(perturbed)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Variance as mass proxy
                var = outputs.var().item()
                masses.append(var)

        return np.mean(masses)


class CommutationCost:
    """
    Measures commutation cost (Eq. 3 from MCC paper).

    μ_s = D_KL[E(P∘S) || E(S∘P)]

    Where:
    - S = semantic evolution (forward pass)
    - P = perturbation
    - E = entropy
    """

    @staticmethod
    def compute(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        perturb_fn: Optional[Callable] = None,
        epsilon: float = 0.1
    ) -> float:
        """
        Compute commutation cost.

        Args:
            model: Model to evaluate
            inputs: Input tensor
            perturb_fn: Optional custom perturbation function
            epsilon: Perturbation magnitude

        Returns:
            Commutation cost (scalar, >= 0)
        """
        if perturb_fn is None:
            perturb_fn = lambda x: x + torch.randn_like(x) * epsilon

        model.eval()

        with torch.no_grad():
            # Path 1: Evolve then perturb (S∘P applied to input, then measure)
            outputs_clean = model(inputs)
            if isinstance(outputs_clean, tuple):
                outputs_clean = outputs_clean[0]
            outputs_perturbed = perturb_fn(outputs_clean)
            entropy_sp = CommutationCost._entropy(outputs_perturbed)

            # Path 2: Perturb then evolve (P∘S)
            inputs_perturbed = perturb_fn(inputs)
            outputs_from_perturbed = model(inputs_perturbed)
            if isinstance(outputs_from_perturbed, tuple):
                outputs_from_perturbed = outputs_from_perturbed[0]
            entropy_ps = CommutationCost._entropy(outputs_from_perturbed)

        # KL divergence between entropy distributions
        # (simplified: absolute difference of mean entropies)
        kl = abs(entropy_sp - entropy_ps)

        return kl

    @staticmethod
    def _entropy(tensor: torch.Tensor) -> float:
        """Compute entropy of tensor treated as distribution."""
        probs = F.softmax(tensor.view(-1), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()

    @staticmethod
    def batch_compute(
        model: torch.nn.Module,
        inputs_list: List[torch.Tensor],
        epsilon: float = 0.1
    ) -> List[float]:
        """Compute commutation cost for multiple inputs."""
        return [
            CommutationCost.compute(model, inp, epsilon=epsilon)
            for inp in inputs_list
        ]


if __name__ == "__main__":
    # Test
    print("Entropy Measurement - Test")
    print("=" * 50)

    # Test entropy tracker
    tracker = EntropyTracker(target_entropy=3.5)
    for e in [1.5, 2.3, 3.8, 4.2, 3.1, 2.9, 4.5]:
        tracker.record(e)
        print(f"Entropy {e:.1f} -> Zone: {tracker.get_zone(e)}")

    print(f"\nZone residence: {tracker.zone_residence()}")
    print(f"Mean entropy: {tracker.mean():.2f} ± {tracker.std():.2f}")
