# SmolVLA + Mixture of Horizons on StackPyramid-v1
 
**EEE 598 Spring 2026 — Final Project**
Comparing baseline SmolVLA against SmolVLA augmented with Mixture of Horizons (MoH) action chunking on the ManiSkill3 `StackPyramid-v1` manipulation task.

![Demo](assets/base_ep046_success.gif)
 
---
 
## Models & Datasets
 
### Trained Checkpoints
| Model | HuggingFace | Description |
|-------|-------------|-------------|
| Baseline SmolVLA | [`ceshank01/smolvla-base-stackpyramid-v4`](https://huggingface.co/ceshank01/smolvla-base-stackpyramid-v4) | 128x128, 20K steps, best baseline |
| SmolVLA + MoH | [`ceshank01/smolvla-moh-stackpyramid-v5`](https://huggingface.co/ceshank01/smolvla-moh-stackpyramid-v5) | 128x128, 20K steps, horizons [10,25,50] |
 
### Datasets
| Dataset | HuggingFace | Description |
|---------|-------------|-------------|
| Training data | [`ceshank01/stack-pyramid-v1-v2`](https://huggingface.co/datasets/ceshank01/stack-pyramid-v1-v2) | 998 episodes, 128x128 RGBD "Pick" |
| Training data | [`ceshank01/stack-pyramid-v1-v5`](https://huggingface.co/datasets/ceshank01/stack-pyramid-v1-v5) | 998 episodes, 128x128 RGBD "Push" |
 
---
 
## Environment Setup
 
### Prerequisites
- Linux machine with NVIDIA GPU (tested on RTX 5070 Ti, A100)
- Conda
- CUDA 12+
### Installation
 
```bash
# 1. Clone this repo
git clone https://github.com/cpadar/smolvla-moh-project.git
cd smolvla-moh-project
 
# 2. Create conda environment
conda env create -f environment.yml
conda activate smolvla-moh
 
# 3. Install LeRobot with SmolVLA support
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[smolvla]"
 
# 4. Patch LeRobot to support smolvla_moh policy type
cd ~/smolvla-moh-project
python scripts/slurm/patch_factory.py
```
 
---
 
## Reproducing Evaluation Results
 
### Evaluate Baseline SmolVLA
 
```bash
cd ~/smolvla-moh-project
conda activate smolvla-moh
 
python scripts/eval/eval_policy.py \
  --model ceshank01/smolvla-base-stackpyramid-v4 \
  --model-type base \
  --num-episodes 50 \
  --save-video \
  --video-dir results/eval_baseline
```
 
### Evaluate SmolVLA + MoH
 
```bash
python scripts/eval/eval_policy.py \
  --model ceshank01/smolvla-moh-stackpyramid-v6 \
  --model-type moh \
  --num-episodes 50 \
  --save-video \
  --video-dir results/eval_moh
```
 
### Watch Videos
 
```bash
# Watch a specific episode
mpv --speed=0.75 results/eval_baseline/base_ep000_fail.mp4
mpv --speed=0.75 results/eval_baseline/base_ep000_success.mp4
```
---

## Reproducing Training
 
### Baseline SmolVLA (Google Colab)
 
Open the Colab notebook: **[SmolVLA Baseline Training](https://colab.research.google.com/drive/18pFyyaDVkg5E_SAB-Sd0XsmQOJsBHRWd?usp=sharing)**
This is view only, make a copy and update username info to use
 
### SmolVLA+MoH (ASU SOL)
 
```bash
conda activate smolvla-moh
cd ~/smolvla-moh-project
git pull
sbatch scripts/slurm/train_moh_full.sh
squeue -u $USER
```

---
 
## References
 
- [Mixture of Horizons in Action Chunking](https://arxiv.org/abs/2511.19433) — Jing et al., 2025
- [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) — HuggingFace LeRobot
- [ManiSkill3](https://maniskill.ai/) — Simulation environment
- [LeRobot](https://github.com/huggingface/lerobot) — Training framework
