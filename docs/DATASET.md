# Data Preparation

PS-MOT uses MOT-style video datasets. The main benchmark layout covers DanceTrack, SportsMOT, JRDB, and EmboTrack.

## Dataset Links

- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- JRDB
- EmboTrack

## Directory Layout

Use `./dataset/` by default, or pass your own root through `--data-root`.

```text
dataset/
├── DanceTrack/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_seqmap.txt
│   ├── val_seqmap.txt
│   └── test_seqmap.txt
├── SportsMOT/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_seqmap.txt
│   ├── val_seqmap.txt
│   └── test_seqmap.txt
├── JRDB/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_seqmap.txt
│   ├── val_seqmap.txt
│   └── test_seqmap.txt
└── EmboTrack/
    ├── train/
    ├── val/
    ├── test/
    ├── train_seqmap.txt
    ├── val_seqmap.txt
    └── test_seqmap.txt
```

For standard MOT-style video datasets, each sequence should contain:

```text
<DATA_ROOT>/<DATASET>/<SPLIT>/<SEQUENCE>/
├── img1/
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

## Pseudo-Label Files

The PS-MOT training config reads pseudo labels from:

```text
<DATA_ROOT>/<DATASET>/train/<SEQUENCE>/gt/sam2_pgt.txt
```

Generate these files with:

```bash
python generate_mot_pseudo_boxes.py \
  --data-root ./dataset \
  --dataset DanceTrack \
  --split train \
  --checkpoint ./sam3.pt \
  --pseudo-label-file sam2_pgt.txt
```

The pseudo-label format is:

```text
frame,id,x,y,w,h,mark,class,quality
```

The last field stores the pseudo-label quality score used by UGL.

## Legacy Conversion Scripts

Some helper scripts from the MOTIP codebase are still kept for compatibility with auxiliary experiments:

- BFT: [tools/gen_bft_gts.py](../tools/gen_bft_gts.py)
- CrowdHuman: [tools/gen_crowdhuman_gts.py](../tools/gen_crowdhuman_gts.py)

They are not part of the main PS-MOT benchmark layout above. Check and edit paths inside these scripts before running them.
