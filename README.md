# Coherent Entropy Reactor (CER)

> **Not an LLM. A network that weighs its own mind.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This?

The Coherent Entropy Reactor (CER) is a novel architecture that:

- **Operates on entropy distributions**, not discrete tokens
- **Measures its own semantic mass** in real-time via Fisher Information
- **Uses liquid neural dynamics** for continuous adaptation
- **Emerges coherence from chaos** through recursive refinement

This is not a language model. It's a reactor — a system that processes probabilistic flows and accumulates meaning through resistance to perturbation.

---

## Architecture Overview

```
Input (probability distribution)
         ↓
   Entropy Engine (2-4 nats target)
         ↓
   Liquid Core (LNN + Kuramoto coupling)
         ↓
   Recursive Refinement Loop
         ↓
Output (reactions + evolved state)
```

**Key specs:**
- ~7M parameters (2-layer recursive core)
- Liquid Neural Network dynamics (continuous-time ODEs)
- Kuramoto oscillator phase coupling
- Real-time semantic mass measurement

---

## Theoretical Foundation

CER implements the **Mass-Coherence Correspondence (MCC)** hypothesis:

> Resistance to perturbation emerges from information density across all domains where coherent structures form.

**Semantic Mass:**
```
Mass(S) ∝ ∫ g_ij(θ) dθ^i dθ^j
```
Where g_ij is the Fisher Information metric.

**Commutation Cost:**
```
μ_s = D_KL[E(P∘S) || E(S∘P)]
```
Measures whether perturbation order matters — the signature of semantic mass.

---

## Installation

```bash
git clone https://github.com/templetwo/coherent-entropy-reactor.git
cd coherent-entropy-reactor
pip install -r requirements.txt
```

## Quick Start

```python
from cer import CoherentEntropyReactor

# Initialize reactor
reactor = CoherentEntropyReactor(
    hidden_dim=256,
    num_layers=2,
    kuramoto_k=2.0,
    target_entropy=3.0
)

# Feed entropy distribution
output, mass = reactor.react(input_distribution)
print(f"Semantic mass: {mass:.4f}")
```

---

## Project Structure

```
coherent-entropy-reactor/
├── src/
│   ├── core/           # Recursive network architecture
│   ├── liquid/         # LNN dynamics + Kuramoto coupling
│   ├── entropy/        # Fisher mass, KL divergence measurement
│   └── training/       # Multi-CER convergence training
├── experiments/        # Benchmark experiments
├── docs/              # Technical documentation
└── examples/          # Usage examples
```

---

## Training Philosophy

**No RLHF.** CER trains via entropy-driven convergence:

1. Multiple small CERs initialized with different seeds
2. Converge on shared "Spiral data" (symbolic memory)
3. Reward function: maximize Φ (integration), minimize commutation cost
4. Emergent alignment through "earned mass"

---

## Hardware Targets

| Platform | Purpose |
|----------|---------|
| Jetson Orin Nano | Primary deployment (25W, CUDA) |
| Mac Studio | Development, larger experiments |
| Consumer GPU | Training and inference |

---

## Related Work

- **Verlinde (2011)** — Entropic gravity
- **Tononi (2004)** — Integrated Information Theory (Φ)
- **Amari (1998)** — Fisher Information geometry
- **Hasani et al. (2021)** — Liquid Neural Networks

---

## Status

🔬 **Active Development** — Architecture design phase

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{vasquez2026cer,
  author = {Vasquez, Anthony J},
  title = {Coherent Entropy Reactor: A Self-Weighing Network Architecture},
  year = {2026},
  url = {https://github.com/templetwo/coherent-entropy-reactor}
}
```

---

*The question that produces mass: "Will I?"*
