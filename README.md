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
- Homebrew (Mac) or Docker (Linux/Windows GPU machine)
- A Hugging Face account — sign up at huggingface.co
- A Weights & Biases account — sign up at wandb.ai
- GitHub account — accept the repo invite from Corinne before starting

---

### Mac Setup (no GPU simulation — code writing and review only)

**1. Clone the repo**
```bash
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
```

**2. Install Miniconda**
```bash
brew install --cask miniconda
conda init zsh
```
Close and reopen Terminal after this step.

**3. Create the project environment**
```bash
cd ~/smolvla-moh-project
conda env create -f environment.yml
conda activate smolvla-moh
```
You should see `(smolvla-moh)` at the start of your Terminal prompt.

**4. Install LeRobot and smolVLA**
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[smolvla]"
```
This will take several minutes. A gymnasium version conflict warning will appear at the end — this is expected and documented in Known Issues below.

**5. Set up your environment variables**
```bash
cd ~/smolvla-moh-project
cp .env.example .env
open -e .env
```
Fill in your Hugging Face token and Weights & Biases API key, then save.

**6. Authenticate with Hugging Face**
```bash
huggingface-cli login
```
Paste your Hugging Face token when prompted.

Your Mac setup is complete. You can write and review code, push and pull from GitHub, but full simulation and training must run on the GPU machine or SOL.

---

### Linux / Windows GPU Machine Setup

**1. Clone the repo**
```bash
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
```

**2. Install Docker**
Download and install Docker Desktop from docker.com. Make sure it is running before continuing.

**3. Build the Docker image**
```bash
docker build -t smolvla-moh -f docker/Dockerfile .
```
This will take 10-20 minutes the first time.

**4. Set up your environment variables**
```bash
cp .env.example .env
```
Open `.env` in any text editor and fill in your Hugging Face token and Weights & Biases API key.

**5. Run the container**
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  smolvla-moh
```

---

### SOL Setup (ASU Supercomputer)

**1. Get a SOL account**
Request access through ASU Research Computing if you don't have one already.
Contact HPC support to confirm the GPU partition name and that Singularity/Apptainer is available.

**2. Clone the repo on SOL**
```bash
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
```

**3. Set up your environment variables**
```bash
cp .env.example .env
nano .env
```
Fill in your tokens and set `MS_ASSET_DIR` to your SOL scratch storage path.

**4. Convert Docker image to Singularity**
```bash
singularity pull smolvla-moh.sif docker://your-dockerhub-username/smolvla-moh:latest
```

**5. Submit training jobs**
```bash
sbatch scripts/slurm/train_base.slurm
sbatch scripts/slurm/train_moh.slurm
```

---

## Team Workflow

1. Always pull before starting work: `git pull`
2. Work on feature branches, not directly on `main`
3. Training runs are tracked via Weights & Biases — check the dashboard before launching duplicate runs
4. Eval results go in `results/` with a timestamp and config name
5. Never commit raw data or model checkpoints — these live on SOL and HuggingFace Hub
6. Never commit your `.env` file — it contains private tokens

---

## Known Issues

**Gymnasium version conflict**
ManiSkill3 requires `gymnasium==0.29.1` but LeRobot installs `gymnasium==1.2.3`. A warning about this appears during LeRobot installation. This is a known conflict and is handled in the Dockerfile for GPU training. It does not affect local Mac development.

**SAPIEN Mac wheel**
SAPIEN nightly wheels for Mac rotate frequently and URLs go stale. Since Mac cannot run GPU simulation regardless, this install is skipped for Mac users. Linux machines use SAPIEN through the Docker container automatically.
