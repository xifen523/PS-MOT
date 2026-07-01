# Tutorial

This page summarizes the main moving parts of PS-MOT and the runtime settings that are often adjusted during experiments.

## Two-Stage Workflow

PS-MOT follows a two-stage point-supervised training pipeline:

1. Generate pseudo boxes from point prompts with SAM 3.
2. Train the tracker with pseudo-label quality scores.

The pseudo-label generator writes `sam2_pgt.txt` files under each sequence's `gt/` directory. The PS-MOT training config uses `DanceTrackS`, which maps to [data/dancetrack_pseudo.py](../data/dancetrack_pseudo.py) and reads those pseudo labels.

## PEWA

Point-Excited Wavelet Attention is implemented in the Deformable DETR backbone path. It uses point information to strengthen high-frequency boundary features for point-supervised instance learning.

Relevant code:

- [models/deformable_detr/backbone.py](../models/deformable_detr/backbone.py)
- [models/deformable_detr/deformable_detr.py](../models/deformable_detr/deformable_detr.py)

## UGL

Uncertainty-Guided Gaussian Learning uses the pseudo-label quality field to calibrate box supervision. The pseudo-label loader stores the last field of `sam2_pgt.txt` as `quality`, and the DETR criterion consumes it during loss computation.

Relevant code:

- [data/dancetrack_pseudo.py](../data/dancetrack_pseudo.py)
- [data/util.py](../data/util.py)
- [models/deformable_detr/deformable_detr.py](../models/deformable_detr/deformable_detr.py)

## Temporal Length

The tracker handles target disappearance and re-appearance within a temporal tolerance window. These parameters should be changed together:

- `SAMPLE_LENGTHS`: sampled video clip length during training.
- `REL_PE_LENGTH`: maximum length of relative temporal position encoding.
- `MISS_TOLERANCE`: tolerance for missing targets during inference.

A practical rule is:

```text
SAMPLE_LENGTHS == REL_PE_LENGTH >= MISS_TOLERANCE
```

## Inference Thresholds

The tracker uses thresholds for detection and association.

Object detection:

- `DET_THRESH`: detections below this confidence are not used for tracking.
- `NEWBORN_THRESH`: unmatched detections become new tracks only above this threshold.

Object association:

- `ID_THRESH`: minimum ID assignment confidence for a valid association.

These values can be set in the YAML config or overridden from the command line:

```bash
--det-thresh 0.3 --newborn-thresh 0.6 --id-thresh 0.2
```

## Memory Usage

If GPU memory is limited, reduce the number of checkpoint frames:

```bash
--detr-num-checkpoint-frames 2
```

For very small GPUs, try `1` and reduce `SAMPLE_LENGTHS` for debugging.
