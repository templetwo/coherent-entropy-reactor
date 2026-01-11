"""
Liquid Neural Network Dynamics + Kuramoto Oscillator

Implements continuous-time adaptive dynamics for the CER.
Based on Liquid Time-Constant Networks (Hasani et al., 2021)
with Kuramoto phase coupling for entropy modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class KuramotoOscillator(nn.Module):
    """
    Kuramoto oscillator for phase-coupled entropy modulation.

    dφ_i/dt = ω_i + (K/N) * Σ sin(φ_j - φ_i)

    Used to modulate sampling temperature:
    T = T_base + A * sin(φ_mean)
    """

    def __init__(
        self,
        n_oscillators: int = 8,
        coupling_strength: float = 2.0,
        natural_frequency: float = 1.0,
        amplitude: float = 0.4,
        base_temperature: float = 0.8
    ):
        super().__init__()

        self.n_oscillators = n_oscillators
        self.K = coupling_strength
        self.amplitude = amplitude
        self.base_temperature = base_temperature

        # Initialize phases uniformly
        self.register_buffer(
            'phases',
            torch.linspace(0, 2 * math.pi, n_oscillators)
        )

        # Natural frequencies (slight variation)
        self.register_buffer(
            'frequencies',
            torch.ones(n_oscillators) * natural_frequency +
            torch.randn(n_oscillators) * 0.1
        )

    def step(self, dt: float = 0.1, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance oscillator phases by dt.

        Args:
            dt: Time step
            external_input: Optional external forcing term

        Returns:
            Mean phase (for temperature modulation)
        """
        # Compute phase differences
        phase_diffs = self.phases.unsqueeze(1) - self.phases.unsqueeze(0)

        # Kuramoto coupling term
        coupling = (self.K / self.n_oscillators) * torch.sin(phase_diffs).sum(dim=1)

        # Phase update
        dphase = self.frequencies + coupling
        if external_input is not None:
            dphase = dphase + external_input.mean() * 0.1

        self.phases = self.phases + dphase * dt

        # Keep phases in [0, 2π]
        self.phases = self.phases % (2 * math.pi)

        return self.mean_phase()

    def mean_phase(self) -> torch.Tensor:
        """Compute mean phase (order parameter angle)."""
        return torch.atan2(
            torch.sin(self.phases).mean(),
            torch.cos(self.phases).mean()
        )

    def order_parameter(self) -> torch.Tensor:
        """
        Compute Kuramoto order parameter R.

        R = |1/N * Σ exp(i*φ_j)|

        R ≈ 1: synchronized
        R ≈ 0: desynchronized
        """
        complex_phases = torch.exp(1j * self.phases.to(torch.complex64))
        return torch.abs(complex_phases.mean())

    def get_temperature(self) -> torch.Tensor:
        """
        Get current temperature modulation.

        T = T_base + A * sin(φ_mean)
        """
        return self.base_temperature + self.amplitude * torch.sin(self.mean_phase())

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature modulation to tensor.

        Args:
            x: Input tensor (logits or hidden state)

        Returns:
            Modulated tensor
        """
        temperature = self.get_temperature()
        return x / temperature


class LiquidCell(nn.Module):
    """
    Liquid Time-Constant Cell.

    Implements ODE: τ(x) * dx/dt = -x + f(x, I)

    Where τ(x) is a learned time constant that adapts to input.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recurrent connection
        self.recurrent = nn.Linear(hidden_dim, hidden_dim)

        # Time constant network (makes τ adaptive)
        self.tau_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # τ in (0, 1)
        )

        # Output activation
        self.activation = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Forward pass with ODE integration.

        Args:
            x: Input [batch, input_dim]
            h: Hidden state [batch, hidden_dim]
            dt: Time step for integration

        Returns:
            Updated hidden state
        """
        batch_size = x.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Compute adaptive time constant
        tau = self.tau_net(h) + 0.1  # Ensure tau > 0

        # Compute state derivative
        input_contrib = self.input_proj(x)
        recurrent_contrib = self.recurrent(h)

        # ODE: τ * dh/dt = -h + tanh(Wx + Uh)
        dh = (-h + self.activation(input_contrib + recurrent_contrib)) / tau

        # Euler integration
        h_new = h + dh * dt

        return h_new


class LiquidLayer(nn.Module):
    """
    Full Liquid Neural Network layer with Kuramoto coupling.

    Combines:
    - LiquidCell for adaptive dynamics
    - KuramotoOscillator for phase modulation
    - Residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_oscillators: int = 8,
        kuramoto_k: float = 2.0,
        integration_steps: int = 5
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.integration_steps = integration_steps

        # Liquid cell
        self.cell = LiquidCell(input_dim, hidden_dim)

        # Kuramoto oscillator
        self.kuramoto = KuramotoOscillator(
            n_oscillators=n_oscillators,
            coupling_strength=kuramoto_k
        )

        # Output projection (to match input dim for residual)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Layer norm
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through liquid dynamics.

        Args:
            x: Input [batch, seq, input_dim]
            h: Initial hidden state

        Returns:
            output: Processed sequence [batch, seq, input_dim]
            h_final: Final hidden state
        """
        batch_size, seq_len, _ = x.shape

        outputs = []
        h_current = h

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Multiple integration steps per token
            for _ in range(self.integration_steps):
                # Step Kuramoto
                self.kuramoto.step(dt=0.1, external_input=x_t)

                # Update liquid cell
                h_current = self.cell(x_t, h_current, dt=0.1)

                # Apply Kuramoto modulation
                h_current = self.kuramoto.modulate(h_current)

            outputs.append(h_current)

        # Stack outputs
        output_seq = torch.stack(outputs, dim=1)

        # Project back to input dim with residual
        output = self.output_proj(output_seq)
        output = self.norm(x + output)

        return output, h_current

    def get_order_parameter(self) -> float:
        """Get current Kuramoto synchronization."""
        return self.kuramoto.order_parameter().item()

    def get_temperature(self) -> float:
        """Get current modulated temperature."""
        return self.kuramoto.get_temperature().item()


if __name__ == "__main__":
    # Test
    print("Liquid Dynamics - Test")
    print("=" * 50)

    # Test Kuramoto
    kuramoto = KuramotoOscillator(n_oscillators=8, coupling_strength=2.0)
    print(f"Initial order parameter: {kuramoto.order_parameter():.4f}")

    for i in range(50):
        kuramoto.step(dt=0.1)

    print(f"After 50 steps: {kuramoto.order_parameter():.4f}")
    print(f"Temperature: {kuramoto.get_temperature():.4f}")

    # Test Liquid Layer
    print("\nLiquid Layer:")
    layer = LiquidLayer(input_dim=64, hidden_dim=128, kuramoto_k=2.0)

    x = torch.randn(4, 16, 64)  # batch=4, seq=16, dim=64
    output, h = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {h.shape}")
    print(f"Order parameter: {layer.get_order_parameter():.4f}")
