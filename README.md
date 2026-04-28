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
├── notebooks/                # Google Colab Notebook for SmolVLA fine-tuning
├── results/                  # Eval output logs and plots (gitignored except structure)
└── environment.yml           # Conda environment (for local dev / SOL)
```

## Quickstart

### Prerequisites
- Git
- A Hugging Face account
- A Weights & Biases account
- GitHub account
- GPU for simulation
- Google Colab access for fine-tuning SmolVLA
- Supercomputer access for fine-tuning SmolVLA+MOH

---

### Linux / Windows GPU Machine Setup

**1. Clone the repo**
```bash
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
```

**2. Create the project environment**
```bash
cd ~/smolvla-moh-project
conda env create -f environment.yml
conda activate smolvla-moh
```
You should see `(smolvla-moh)` at the start of your Terminal prompt.

**3. Install LeRobot and smolVLA**
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[smolvla]"
```

---

### SOL Setup (ASU Supercomputer)

**1. Clone the repo on SOL**
```bash
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
```

**2. Create the project environment**
```bash
cd ~/smolvla-moh-project
conda env create -f environment.yml
conda activate smolvla-moh
```
You should see `(smolvla-moh)` at the start of your Terminal prompt.

**3. Submit training jobs**
```bash
sbatch scripts/slurm/train_base.slurm
sbatch scripts/slurm/train_moh.slurm
```

---
## Model Fine-tuning 
SmolVLA can be fine-tuned on Google Colab notebook. See link to documentation on how to setup training. 
https://huggingface.co/docs/lerobot/smolvla#finetune-smolvla-on-your-data

To ensure fair training between models SmolVLA+MOH was fine-tuned on ASU SOL due to Google Colab limitations for the bigger model. 


## Known Issues

**Gymnasium version conflict**
ManiSkill3 requires `gymnasium==0.29.1` but LeRobot installs `gymnasium==1.2.3`. A warning about this appears during LeRobot installation.
