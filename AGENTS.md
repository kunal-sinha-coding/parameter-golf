# Agent Notes

- MLX training scripts (`train_gpt_mlx.py`, `train_gpt_mlx_ar.py`) must be run outside the default sandbox. In the sandbox, `import mlx.core` can crash during Metal device initialization because the process cannot access the Apple GPU.
- When running those scripts from Codex, request escalated permissions for the exact Python command so MLX can access Metal normally.
- The validated smoke-test command is:

```sh
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```
