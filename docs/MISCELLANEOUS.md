# Miscellaneous

This page collects runtime settings that are useful when reproducing experiments.

## Logging

WandB logging is disabled by default in the provided configs:

```yaml
USE_WANDB: False
```

To enable WandB:

```bash
accelerate launch --num_processes=8 train.py \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml \
  --use-wandb True \
  --exp-name <experiment_name>
```

Set `EXP_OWNER`, `EXP_PROJECT`, and `EXP_GROUP` in the YAML file or override them with runtime arguments when needed.

## Checkpoints

Training checkpoints are written under `OUTPUTS_DIR` and `EXP_NAME`. If `OUTPUTS_DIR` is not set, the training script creates an output directory from the experiment settings.

Useful checkpoint-related options:

- `--resume-model`: resume from a saved checkpoint.
- `--resume-optimizer`: resume optimizer state.
- `--resume-scheduler`: resume scheduler state.
- `--save-checkpoint-per-epoch`: checkpoint saving interval.

## Reproducibility

Set the seed in the config or with:

```bash
--seed 42
```

Record the exact config file, command line, and checkpoint used for each experiment. You can also pass a revision tag through:

```bash
--git-version <revision>
```
