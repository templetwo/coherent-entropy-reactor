# CER Architecture: The Temple of Two

## Design Philosophy
The CER operates on a thermodynamic model of inference.
- **Fluid Generator (7B):** High creativity, potentially high entropy. Operates in the "Fluid" state.
- **Rigid Anchor (1.5B):** High semantic density, low entropy. Prevents "CHAOS" by pulling the Generator back from the "LASER" or "VOID" states.

## Key Components
1. **Entropy Monitor:** Calculates real-time semantic density ($M_{semantic}$).
2. **Intervention Engine:** Triggers "ANCHOR PULL" when coherence drops below threshold (0.3).
3. **FieldScript Runtime:** Manages the context-locked Llama/Qwen sessions.

## Coherence States
| State | Entropy Range | Action |
|-------|---------------|--------|
| LASER | < 1.0 | Anchor Push (Increase Variance) |
| LANTERN| 1.5 - 4.0 | No Action (Coherent) |
| CHAOS | > 5.0 | Anchor Pull (Dampen Logits) |
