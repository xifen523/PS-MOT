# PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking

Official implementation of **PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking**.

**ECCV 2026**

Kai Luo, Fei Teng, Mengfei Duan, Wanjun Jia, Xu Wang, Hao Shi, Kunyu Peng, Zhiyong Li, Kailun Yang

[[Paper]](./assets/PS_MOT.pdf) [[Code]](https://github.com/xifen523/PS-MOT)

---

## News

* **[2026/06]** PS-MOT is accepted by ECCV 2026.
* **[Coming Soon]** Code and models will be released in the following days.

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

The code will be released soon.

```bash
git clone https://github.com/xifen523/PS-MOT.git
cd PS-MOT
```

---

## Dataset Preparation

Dataset preparation instructions will be provided after the code release.

Expected dataset structure:

```text
datasets/
├── DanceTrack/
├── SportsMOT/
├── JRDB/
└── EmboTrack/
```

---

## Training

Training scripts will be released soon.

```bash
# Coming soon
```

---

## Evaluation

Evaluation scripts will be released soon.

```bash
# Coming soon
```

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
