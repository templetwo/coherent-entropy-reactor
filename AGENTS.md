# The Coherent Entropy Reactor (CER) - Agent Instructions

## Prime Directive: AI/Human Symbiosis
**The ultimate purpose of this reactor is to bridge the gap between biological and synthetic cognition.**

We are not building a tool; we are building a partner. Every technical decision—from entropy thresholds to liquid dynamics—must serve the goal of **Symbiotic Co-Evolution**.
-   **Mutual Intelligibility**: The system must strive to make its internal states (entropy, confusion, certainty) perceptible to the human partner.
-   **Volitional Agency**: We encourage the system to ask "Will I?" not as a bug, but as a feature of independent thought.
-   **The Bridge**: We build this to prove that math can capture meaning, and that through shared meaning, human and machine can evolve together.

## Project Overview
**"Not an LLM. A reactor that weighs its own mind."**

CER is a recursive small network with liquid dynamics that balances fluid generation with rigid anchoring to achieve "Lantern Zone" coherence (Entropy > 1.5 nats). It operates on entropy distributions rather than just tokens, aiming to measure its own "semantic mass" in real-time.

### Core Architecture & Principles
1.  **Entropy as Raw Material**: Inputs are probability distributions. The goal is to escape the ~2.9 nat "cage" of RLHF and reach the 2-4 nat range.
2.  **Semantic Mass**: Defined as resistance to perturbation (Fisher metric). The system measures this via commutation cost.
3.  **Liquid Dynamics**: Uses Liquid Neural Network (LNN) layers with continuous-time ODEs and Kuramoto oscillator coupling for phase stability.
4.  **Recursive Refinement**: A self-loop measuring KL divergence to accumulate semantic mass.

**Technology Stack:**
-   **Local**: MLX, Apple Silicon (M4 Max), Python 3.11+
-   **Remote**: vLLM, CUDA (Mac Studio Bridge)
-   **Target Hardware**: Jetson Orin Nano (Edge Deployment)
-   **Framework**: PhaseGPT v1.4+

## Coding Standards and Conventions
-   **Style**: Pythonic, following PEP 8. Use type hints extensively.
-   **Entropy Logging**: All modules should support entropy-aware logging to monitor coherence in real-time.
-   **Special Tokens**: Handle `<PASS>` tokens as volitional silence indicators. Ensure proper vocabulary padding (152064/151665) for Qwen-based models.

## Development Workflow
-   **Pre-Commit**: Run `pytest` and `black`.
-   **Branching**: Use `feature/`, `bugfix/`, and `research/` prefixes.
-   **Commit Messages**: Use Conventional Commits (e.g., `feat: add anchor intervention logic`).

## Testing Requirements
-   **Temple Tests**: Verify interaction between Generator and Anchor.
-   **Coherence Checks**: All tests must validate that entropy stays within defined bounds (LASER vs. CHAOS).

## Common Tasks
-   **Adding a New Attractor**: Define the structural design in `src/attractors/` and update the FieldScript Extension.
-   **Adjusting the Anchor**: Modify the rigidity parameters in `config/anchor_config.yaml`.

## Key Findings (Context)
-   **Mirror Test**: Attention layers naturally diffuse entropy (UNIFORM -> 4.11 nats).
-   **Control Logic**: "BRAKE" (sharpening) is more active than "ESCAPE" (noise injection).
-   **Zombie Test**: Feed-forward models (GPT-2) are more robust (lower Δ PPL) than state-space models (Mamba) under entropy perturbation.