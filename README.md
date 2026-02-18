# SmolVLA + Mixture of Horizons — StackPyramid-v1

A research project comparing baseline SmolVLA against SmolVLA augmented with
Mixture of Horizons (MoH) action chunking on the ManiSkill3 `StackPyramid-v1` task.

## Project Overview

| | Baseline | MoH |
|---|---|---|
| Model | SmolVLA (450M) | SmolVLA + Mixture of Horizons |
| Task | StackPyramid-v1 | StackPyramid-v1 |
| Training data | ManiSkill3 dataset | same |
| Eval | Randomized cube positions (fixed seeds) | same seeds |

## Repo Structure

```
smolvla-moh-project/
├── configs/                  # YAML configs for training and eval runs
│   ├── train_base.yaml
│   ├── train_moh.yaml
│   └── eval.yaml
├── data/                     # Data loading utilities (not raw data)
│   └── maniskill_dataset.py
├── models/
│   ├── smolvla_base/         # Fine-tuning wrapper for baseline SmolVLA
│   └── smolvla_moh/          # SmolVLA + MoH modifications
├── scripts/
│   ├── train/                # Training entry points
│   ├── eval/                 # Evaluation entry points
│   └── slurm/                # SLURM batch scripts for SOL supercomputer
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
├── notebooks/                # Exploratory notebooks
├── results/                  # Eval output logs and plots (gitignored except structure)
└── environment.yml           # Conda environment (for local dev / SOL)
```

## Quickstart

### Prerequisites

- Git
- Docker (for GPU teammates and SOL via Singularity)
- OR: Conda (for direct local setup)
- A Hugging Face account (for dataset and model access)

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/smolvla-moh-project.git
cd smolvla-moh-project
```

### 2. Set up your environment

**Option A — Docker (recommended for GPU machines):**
```bash
docker build -t smolvla-moh -f docker/Dockerfile .
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  smolvla-moh
```

**Option B — Conda (Mac / local dev without full GPU sim):**
```bash
conda env create -f environment.yml
conda activate smolvla-moh

# Mac only: install nightly SAPIEN wheel (not yet on PyPI)
PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
pip install https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-$PY_VERSION-$PY_VERSION-macosx_12_0_universal2.whl

# All platforms: install LeRobot + smolVLA extras
git clone https://github.com/huggingface/lerobot.git /opt/lerobot
cd /opt/lerobot && pip install -e ".[smolvla]"
```
> Note: On Mac, ManiSkill3 runs CPU-only simulation (GPU sim is not yet supported on macOS). Full training must happen on a CUDA GPU machine or SOL.

### 3. Authenticate with Hugging Face

```bash
huggingface-cli login
```

### 4. Download ManiSkill3 assets

```bash
python -m mani_skill.utils.download_asset StackPyramid-v1
```

### 5. Running on SOL (ASU Supercomputer)

See `scripts/slurm/` for job submission scripts. Convert the Docker image to Singularity first:
```bash
singularity pull smolvla-moh.sif docker://your-dockerhub-username/smolvla-moh:latest
```
Then submit jobs with:
```bash
sbatch scripts/slurm/train_base.slurm
sbatch scripts/slurm/train_moh.slurm
```

## Team Workflow

1. Always pull before starting work: `git pull`
2. Work on feature branches, not directly on `main`
3. Training runs are tracked via Weights & Biases — check the project dashboard before launching duplicate runs
4. Eval results go in `results/` with a timestamp and config name
5. Never commit raw data or model checkpoints — these live on SOL and HuggingFace Hub

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
MS_ASSET_DIR=/path/to/maniskill/data
```

## Known Dependency Conflicts
ManiSkill3 requires gymnasium 0.29.1 but LeRobot installs 1.2.3, and that this is handled in the Dockerfile.
