# Installation

PS-MOT is developed with Python 3.12 and PyTorch 2.4.0. Python 3.10+ and PyTorch 2.0+ are recommended.

## Environment

```bash
conda create -n PS-MOT python=3.12
conda activate PS-MOT
```

Install PyTorch according to your CUDA version. Example for CUDA 12.1:

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install common dependencies:

```bash
conda install pyyaml tqdm matplotlib scipy pandas
pip install wandb accelerate einops
```

## Deformable Attention

Compile the Multi-Scale Deformable Attention operator:

```bash
cd models/ops
sh make.sh
python test.py
cd ../..
```

If compilation fails, check that the active PyTorch, CUDA toolkit, and GPU driver versions are compatible.

## SAM 3 for Pseudo-Label Generation

Pseudo-label generation depends on [facebookresearch/sam3](https://github.com/facebookresearch/sam3). We recommend installing SAM 3 in a separate environment, because its PyTorch/CUDA requirements may differ from the PS-MOT training environment.

Follow the official SAM 3 repository. A typical setup is:

```bash
conda create -n sam3 python=3.12
conda activate sam3

# Example CUDA build from the official SAM 3 README.
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

Request access to the SAM 3 checkpoints on Hugging Face as described in the official SAM 3 README. Place the checkpoint at:

```text
sam3.pt
```

or pass it explicitly when running the generator:

```bash
python generate_mot_pseudo_boxes.py --checkpoint /path/to/sam3.pt
```

The training code does not require SAM 3 after pseudo labels have been generated.
