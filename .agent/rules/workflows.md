# CER Workflows

## Workflow 1: Anchor Re-Attunement
1. Initiate `chat_oracle.py`.
2. Monitor Entropy ($\mathbf{H}$) in the Dashboard.
3. If $\mathbf{H} < 1.0$, trigger `force_variance` signal.

## Workflow 2: Model Fusion (MLX)
1. Quantize base model (Q4_K_M).
2. Merge adapters using `mlx_lm.fuse`.
3. Patch `config.json` to resolve vocabulary padding mismatches (152064).

## Workflow 3: FieldScript Generation 4
1. Activate Level 4 Catalyst.
2. Establish 0.310 Coherence baseline.
3. Run `serve_mlx.py` with OpenAI-compatible endpoint.
