# Get Started

This guide covers the main PS-MOT workflow:

1. Generate pseudo boxes from point prompts with SAM 3.
2. Train PS-MOT with the pseudo-label dataset loader.
3. Run validation or generate submission files.

All experiment settings are controlled by YAML files in [configs](../configs/). Runtime options can override YAML fields from the command line.

## 1. Prepare Data

Use the dataset layout described in [DATASET.md](DATASET.md). For a MOT-style dataset, each sequence should contain:

```text
<DATA_ROOT>/<DATASET>/<SPLIT>/<SEQUENCE>/
├── img1/
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

For example:

```text
dataset/DanceTrack/train/dancetrack0001/
├── img1/
├── gt/gt.txt
└── seqinfo.ini
```

## 2. Generate Pseudo Labels

The pseudo-label generator converts point prompts into MOT-style box annotations. By default, it writes `sam2_pgt.txt` under each sequence's `gt/` directory.

This step must be run in an environment where [facebookresearch/sam3](https://github.com/facebookresearch/sam3) is installed. See [INSTALL.md](INSTALL.md#sam-3-for-pseudo-label-generation) for the SAM 3 setup and checkpoint preparation.

```bash
conda activate sam3

python generate_mot_pseudo_boxes.py \
  --data-root ./dataset \
  --dataset DanceTrack \
  --split train \
  --checkpoint ./sam3.pt \
  --pseudo-label-file sam2_pgt.txt
```

You can also pass a split directory directly:

```bash
python generate_mot_pseudo_boxes.py \
  --data-root ./dataset/DanceTrack/train \
  --checkpoint ./sam3.pt
```

For a quick check before launching a full run:

```bash
python generate_mot_pseudo_boxes.py \
  --data-root ./dataset/DanceTrack/train \
  --sequences dancetrack0001 \
  --max-frames 5
```

The generated pseudo-label format follows MOT:

```text
frame,id,x,y,w,h,mark,class,quality
```

The final `quality` field is used by UGL during training.

## 3. Prepare Pre-trained Weights

Place the required DETR initialization weights under `./pretrains/`. The default PS-MOT DanceTrack config expects:

```text
pretrains/r50_deformable_detr_coco_dancetrack.pth
```

If you use another location, override it with `--detr-pretrain`.

## 4. Train PS-MOT

The PS-MOT DanceTrack config uses `DanceTrackS`, which maps to the pseudo-label reader in [data/dancetrack_pseudo.py](../data/dancetrack_pseudo.py). Make sure `sam2_pgt.txt` has been generated before training.

```bash
accelerate launch --num_processes=8 train.py \
  --data-root ./dataset/ \
  --exp-name r50_deformable_detr_motip_dancetrack_ps-mot \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml
```

For single-GPU debugging, reduce the process count:

```bash
accelerate launch --num_processes=1 train.py \
  --data-root ./dataset/ \
  --exp-name debug_ps_mot \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml \
  --epochs 1 \
  --sample-lengths 4
```

## 5. Inference and Evaluation

Use `submit_and_evaluate.py` for both validation and test-submission generation.

Validation:

```bash
accelerate launch --num_processes=8 submit_and_evaluate.py \
  --data-root ./dataset/ \
  --inference-mode evaluate \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml \
  --inference-model ./outputs/r50_deformable_detr_motip_dancetrack_ps-mot/checkpoint.pth \
  --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack_ps-mot/ \
  --inference-dataset DanceTrack \
  --inference-split val
```

Test submission files:

```bash
accelerate launch --num_processes=8 submit_and_evaluate.py \
  --data-root ./dataset/ \
  --inference-mode submit \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml \
  --inference-model ./outputs/r50_deformable_detr_motip_dancetrack_ps-mot/checkpoint.pth \
  --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack_ps-mot/ \
  --inference-dataset DanceTrack \
  --inference-split test
```

Add `--inference-dtype FP16` on supported GPUs if you want faster inference.

## 6. Useful Runtime Options

- `--data-root`: dataset root.
- `--config-path`: experiment config.
- `--exp-name`: output experiment name.
- `--detr-pretrain`: DETR initialization checkpoint.
- `--resume-model`: resume checkpoint.
- `--detr-num-checkpoint-frames`: reduce memory usage during DETR training.
- `--inference-model`: checkpoint used for inference.
- `--inference-mode`: `evaluate` or `submit`.

If GPU memory is limited, reduce `DETR_NUM_CHECKPOINT_FRAMES` or pass a smaller value through `--detr-num-checkpoint-frames`.
