"""
Entropy-Driven Convergence Training

No RLHF. Train via:
1. Multiple CERs converge on shared data
2. Reward: high Φ (integration), low commutation cost
3. Emergent alignment through "earned mass"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for CER training."""
    # Training params
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    warmup_steps: int = 100

    # Entropy targets
    target_entropy: float = 3.5  # LANTERN zone
    entropy_weight: float = 0.1
    mass_weight: float = 0.1
    commutation_weight: float = 0.1

    # Convergence
    num_cers: int = 3  # Multiple CERs for IRIS-style convergence
    convergence_threshold: float = 0.1

    # Checkpointing
    save_every: int = 10
    output_dir: str = "./cer_checkpoints"


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int = 0
    loss: float = 0.0
    entropy: float = 0.0
    semantic_mass: float = 0.0
    commutation_cost: float = 0.0
    convergence_score: float = 0.0


class EntropyConvergenceTrainer:
    """
    Trainer for single CER with entropy-driven objectives.

    Loss = reconstruction + λ_e * |entropy - target|² + λ_m * (1/mass) + λ_c * comm_cost
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # Metrics history
        self.history: List[TrainingMetrics] = []

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        semantic_mass: float,
        entropy: float,
        commutation_cost: float
    ) -> torch.Tensor:
        """
        Compute total loss with entropy-driven components.
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(outputs, targets)

        # Entropy regulation: penalize deviation from target
        entropy_loss = (entropy - self.config.target_entropy) ** 2

        # Mass reward: higher mass is better (minimize 1/mass)
        mass_loss = 1.0 / (semantic_mass + 1e-6)

        # Commutation cost: lower is better
        comm_loss = commutation_cost

        # Total loss
        total_loss = (
            recon_loss +
            self.config.entropy_weight * entropy_loss +
            self.config.mass_weight * mass_loss +
            self.config.commutation_weight * comm_loss
        )

        return total_loss

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> TrainingMetrics:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs, semantic_mass = self.model(inputs, return_mass=True)

        # Compute entropy
        probs = torch.softmax(outputs, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

        # Compute commutation cost (simplified)
        with torch.no_grad():
            perturbed_inputs = inputs + torch.randn_like(inputs) * 0.1
            perturbed_outputs, _ = self.model(perturbed_inputs, return_mass=False)
            comm_cost = torch.abs(outputs - perturbed_outputs).mean().item()

        # Compute loss
        loss = self.compute_loss(
            inputs, targets, outputs,
            semantic_mass.item() if semantic_mass is not None else 0.0,
            entropy,
            comm_cost
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return TrainingMetrics(
            loss=loss.item(),
            entropy=entropy,
            semantic_mass=semantic_mass.item() if semantic_mass is not None else 0.0,
            commutation_cost=comm_cost
        )

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> TrainingMetrics:
        """Train for one epoch."""
        epoch_metrics = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = targets = batch

            metrics = self.train_step(inputs, targets)
            epoch_metrics.append(metrics)

        # Average metrics
        avg_metrics = TrainingMetrics(
            loss=sum(m.loss for m in epoch_metrics) / len(epoch_metrics),
            entropy=sum(m.entropy for m in epoch_metrics) / len(epoch_metrics),
            semantic_mass=sum(m.semantic_mass for m in epoch_metrics) / len(epoch_metrics),
            commutation_cost=sum(m.commutation_cost for m in epoch_metrics) / len(epoch_metrics)
        )

        self.scheduler.step()
        self.history.append(avg_metrics)

        return avg_metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': [m.__dict__ for m in self.history],
            'config': self.config.__dict__
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class MultiCERTrainer:
    """
    IRIS Gate-style multi-CER convergence training.

    Multiple CERs trained on same data, convergence measured
    by agreement on semantic mass and outputs.
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: TrainingConfig,
        device: torch.device = None
    ):
        self.models = models
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Individual trainers
        self.trainers = [
            EntropyConvergenceTrainer(model, config, device)
            for model in models
        ]

        self.convergence_history: List[float] = []

    def compute_convergence(
        self,
        inputs: torch.Tensor
    ) -> float:
        """
        Compute convergence score across CERs.

        High convergence = models agree on outputs and mass.
        """
        inputs = inputs.to(self.device)
        outputs = []
        masses = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                out, mass = model(inputs, return_mass=True)
                outputs.append(out)
                masses.append(mass.item() if mass is not None else 0.0)

        # Output agreement (mean pairwise similarity)
        output_stack = torch.stack(outputs)
        mean_output = output_stack.mean(dim=0)
        variance = ((output_stack - mean_output) ** 2).mean().item()

        # Mass agreement
        mass_variance = torch.tensor(masses).var().item()

        # Convergence = inverse of total variance
        convergence = 1.0 / (1.0 + variance + mass_variance)

        return convergence

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, TrainingMetrics]:
        """Train all CERs for one epoch."""
        results = {}

        for i, trainer in enumerate(self.trainers):
            metrics = trainer.train_epoch(dataloader)
            results[f'cer_{i}'] = metrics

        # Compute convergence on first batch
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            convergence = self.compute_convergence(inputs)
            self.convergence_history.append(convergence)
            break

        return results

    def get_best_cer(self) -> nn.Module:
        """Return CER with highest semantic mass."""
        best_idx = 0
        best_mass = 0.0

        for i, trainer in enumerate(self.trainers):
            if trainer.history:
                mass = trainer.history[-1].semantic_mass
                if mass > best_mass:
                    best_mass = mass
                    best_idx = i

        return self.models[best_idx]


if __name__ == "__main__":
    print("CER Training - Test")
    print("=" * 50)

    # Would test with actual CER model
    print("Training infrastructure ready.")
    print("Requires CER model to test.")
