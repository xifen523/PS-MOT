<!-- # PS-MOT
The official implementation of PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking (ECCV 2026) -->

<p align="center">
<h1 align="center"><strong>PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking</strong></h1>
<h3 align="center">ECCV 2026</h3>

Kai Luo, Fei Teng, Mengfei Duan, Wanjun Jia, Xu Wang, Hao Shi, Kunyu Peng, Zhiyong Li, [Kailun Yang](https://yangkailun.com)

[[Paper]](https://arxiv.org/pdf/2606.30476) [[Code]](https://github.com/xifen523/PS-MOT)

---

## News

* **[2026/06]** PS-MOT is accepted by ECCV 2026.
* **[2026/06]** Training, inference, and pseudo-label generation code is now available.

---

## Introduction

Multi-Object Tracking (MOT) has achieved remarkable progress in recent years. However, most existing MOT methods still rely on dense bounding box annotations, which are expensive, labor-intensive, and difficult to scale to large-scale dynamic scenes.

In this work, we introduce **Point-supervised Multi-Object Tracking (PS-MOT)**, a cost-effective alternative to traditional bounding box supervision. Instead of requiring dense bounding boxes, PS-MOT only uses simple point annotations and shifts the supervision paradigm from spatial fitting to topological center-driven representation.

To address the spatial ambiguity and identity drift caused by point supervision, we propose **PS-Track**, a hierarchical framework that progressively cultivates instance awareness from sparse point seeds.

---

## Demo

https://github.com/user-attachments/assets/16363cd3-86d8-4b41-bcca-c4582e7df322

---

## Method

Our PS-Track framework follows a coarse-to-fine point-to-instance evolution paradigm across three levels:

* **Temporal-Feedback Prompting (TFP)** evolves sparse point annotations into temporally consistent pseudo-labels with negative spatial cues and motion priors.
* **Point-Excited Wavelet Attention (PEWA)** activates high-frequency boundary cues from point seeds to enhance instance-aware feature learning.
* **Uncertainty-Guided Gaussian Learning (UGL)** models pseudo-labels as probabilistic distributions and dynamically calibrates supervision intensity.

<p align="center">
  <img src="assets/pipeline2.1.png" width="95%">
</p>

<p align="center">
  <em>Overview of the proposed PS-Track framework.</em>
</p>

---

## Results

PS-Track is evaluated on multiple challenging MOT benchmarks, including:

* **DanceTrack**
* **SportsMOT**
* **JRDB**
* **EmboTrack**

Extensive experiments show that PS-Track achieves competitive performance against fully supervised MOT methods while using only point-level annotations.

---

## Installation

```bash
git clone https://github.com/xifen523/PS-MOT.git
cd PS-MOT
```

The codebase is developed with Python 3.12 and PyTorch 2.4.0. Python 3.10+ and PyTorch 2.0+ are recommended because the project uses newer Python typing features and recent attention-mask behavior in PyTorch.

```bash
conda create -n PS-MOT python=3.12
conda activate PS-MOT

# Install PyTorch according to your CUDA version. Example for CUDA 12.1:
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Common dependencies:
conda install pyyaml tqdm matplotlib scipy pandas
pip install wandb accelerate einops

# Compile Multi-Scale Deformable Attention:
cd models/ops
sh make.sh
python test.py
cd ../..
```

For more details, please refer to [docs/INSTALL.md](docs/INSTALL.md).

---

## Dataset Preparation

Expected dataset structure:

```text
dataset/
├── DanceTrack/
│   ├── train/
│   ├── val/
│   └── test/
├── SportsMOT/
│   ├── train/
│   ├── val/
│   └── test/
├── JRDB/
│   ├── train/
│   ├── val/
│   └── test/
└── EmboTrack/
    ├── train/
    ├── val/
    └── test/
```

Datasets should follow the standard MOT-style layout, where each sequence contains `img1/`, `gt/gt.txt`, and `seqinfo.ini`. See [docs/DATASET.md](docs/DATASET.md) for details.

If you keep datasets outside this repository, pass the location through `--data-root` when running training or inference.

---

## Pseudo-Label Generation

PS-MOT first converts point supervision into MOT-style pseudo boxes. The generator writes one pseudo-label file under each sequence's `gt/` directory. By default, the output file is `sam2_pgt.txt`, which is the file read by the pseudo-label dataset loader.

This step depends on [facebookresearch/sam3](https://github.com/facebookresearch/sam3). We recommend using a separate SAM 3 environment for pseudo-label generation, then returning to the PS-MOT environment for training.

Install SAM 3 following the official repository:

```bash
conda create -n sam3 python=3.12
conda activate sam3

# Example CUDA build from the SAM 3 README. Adjust it to your CUDA/PyTorch setup if needed.
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

Request access to the SAM 3 checkpoints on Hugging Face as described in the official SAM 3 README, then place the checkpoint at `./sam3.pt` in this repository or pass its path through `--checkpoint`.

Run the generator from the PS-MOT repository with the `sam3` environment active:

```bash
conda activate sam3
cd /path/to/PS-MOT

python generate_mot_pseudo_boxes.py \
  --data-root ./dataset \
  --dataset DanceTrack \
  --split train \
  --checkpoint ./sam3.pt \
  --pseudo-label-file sam2_pgt.txt
```

The script also accepts a split directory directly:

```bash
python generate_mot_pseudo_boxes.py \
  --data-root ./dataset/DanceTrack/train \
  --checkpoint ./sam3.pt
```

For quick debugging, use `--sequences` and `--max-frames`:

```bash
python generate_mot_pseudo_boxes.py \
  --data-root ./dataset/DanceTrack/train \
  --sequences dancetrack0001 \
  --max-frames 5
```

The pseudo-label config currently uses `DanceTrackS`, which maps to the pseudo-label reader in [data/dancetrack_pseudo.py](data/dancetrack_pseudo.py).

---

## Training

Put the required DETR pre-trained weights under `./pretrains/`. For DanceTrack PS-MOT training, the config expects:

```bash
./pretrains/r50_deformable_detr_coco_dancetrack.pth
```

Train PS-MOT on pseudo labels:

```bash
accelerate launch --num_processes=8 train.py \
  --data-root ./dataset/ \
  --exp-name r50_deformable_detr_motip_dancetrack_ps-mot \
  --config-path ./configs/r50_deformable_detr_motip_dancetrack_ps-mot.yaml
```

For DETR pre-training or fully supervised MOTIP baselines, use the corresponding configs in [configs](configs/), for example:

```bash
accelerate launch --num_processes=8 train.py \
  --data-root ./dataset/ \
  --exp-name pretrain_r50_deformable_detr_dancetrack \
  --config-path ./configs/pretrain_r50_deformable_detr_dancetrack.yaml
```

If GPU memory is limited, reduce `--detr-num-checkpoint-frames` as described in [docs/GET_STARTED.md](docs/GET_STARTED.md).

---

## Evaluation

Use `submit_and_evaluate.py` for both validation evaluation and test-set submission file generation.

Evaluation on a validation split:

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

Generate tracker files for a test split:

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

You can add `--inference-dtype FP16` for faster inference on supported GPUs.

More detailed training and inference examples are available in [docs/GET_STARTED.md](docs/GET_STARTED.md).

---

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@inproceedings{luo2026psmot,
  title     = {PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking},
  author    = {Luo, Kai and Teng, Fei and Duan, Mengfei and Jia, Wanjun and Wang, Xu and Shi, Hao and Peng, Kunyu and Li, Zhiyong and Yang, Kailun},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

---

## Acknowledgement

This repository is heavily built upon [SAM](https://github.com/facebookresearch/segment-anything) and [MOTIP](https://github.com/MCG-NJU/MOTIP). We sincerely thank the authors for their excellent work and for making their code publicly available.
