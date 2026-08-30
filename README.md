---
title: Oncology Diagnosis CNN
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# oncology-diagnosis-cnn

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)

> A CNN-based skin lesion classifier fine-tuned on HAM10000, reporting the
> clinically-actionable metric (malignant-class sensitivity) that generic
> accuracy numbers hide.

A **CNN (Convolutional Neural Network)** — specifically EfficientNet-B0 — fine-tuned
on **HAM10000**, the standard dermatoscopic imaging benchmark for skin cancer
diagnosis. Classifies a lesion image into one
of seven diagnostic categories and flags the three that are malignant or
pre-malignant.

**Live demo:** upload a dermatoscopic image and get per-class probabilities,
with malignant classes clearly flagged.

## Why this project

Most public skin-lesion classifiers stop at "here's the accuracy." This one is
built the way an oncology screening tool actually needs to be evaluated:
alongside overall accuracy, it reports **sensitivity on the malignant classes
specifically** (`evaluate.py`) — because a model that's 95% accurate overall
but misses melanomas is not a usable screening aid. It also uses **lesion-level
stratified splitting** rather than naive random image splitting, since
HAM10000 contains repeat photographs of the same lesion; splitting by image
instead of by lesion silently leaks information between train and test sets
and inflates reported accuracy.

## The dataset

**HAM10000** ("Human Against Machine with 10000 training images"), Tschandl et
al., 2018 — 10,015 dermatoscopic images across seven diagnostic categories,
each label verified by histopathology, expert consensus, confocal microscopy,
or follow-up.

| Class | Meaning | Malignant? |
|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | Pre-malignant |
| `bcc` | Basal cell carcinoma | Malignant |
| `bkl` | Benign keratosis-like lesion | No |
| `df` | Dermatofibroma | No |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic nevus (common mole) | No |
| `vasc` | Vascular lesion | No |

The dataset is heavily imbalanced (`nv` alone is ~67% of all images; `df` and
`vasc` are each under 2%), which the training pipeline addresses with
inverse-frequency class weighting in the loss function rather than naive
oversampling.

## Clinical context

> Most public skin-lesion classifiers stop at "here's the accuracy." Here's what's different about this repo:

| Common tutorial | This repo |
|---|---|
| Random image-level train/test split | **Lesion-level** stratified split (HAM10000 has repeat photos of the same lesion; splitting by image leaks information and inflates reported accuracy) |
| Report overall accuracy only | Report **malignant-class sensitivity, specificity, and false-negative count** — the numbers that matter for a screening tool |
| Naive oversampling for imbalance | Inverse-frequency **class-weighted loss** |
| No production path | Auto GPU-vs-CPU config presets, mixed precision, honest demo-mode fallback |
| Accuracy only | Full 7-class report + macro ROC-AUC + binary malignant/benign confusion matrix |

Dataset citation:
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions. *Sci Data* 5, 180161 (2018). https://doi.org/10.7910/DVN/DBW86T

**The dataset is not bundled in this repo** (it's ~2.7GB). Download it
yourself from the Harvard Dataverse link above — see Setup below.

## Architecture

EfficientNet-B0 (ImageNet-pretrained) with the classification head replaced by
a small dropout → linear → ReLU → dropout → linear stack fine-tuned on
HAM10000. This is a **classification CNN**, not a U-Net: the task here is
"which of 7 categories is this lesion," a whole-image decision, not
pixel-by-pixel segmentation (that's what the U-Net in
[brain-tumor-segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation)
is for). EfficientNet-B0 was chosen over heavier backbones (ResNet50,
EfficientNet-B4+) as a practical default — it trains in a reasonable time on a
single consumer GPU and is small enough to serve cheaply in a Hugging Face
Space, while remaining competitive with larger architectures on this
particular dataset once fine-tuned (see Expected Performance below).

## Repository structure

```
oncology-diagnosis-cnn/
├── app.py                 # Gradio demo (Hugging Face Spaces entry point)
├── data_prep.py            # Verifies a downloaded dataset is laid out correctly
├── requirements.txt
├── configs/
│   ├── cpu.json             # No GPU: small batch, no AMP -- pipeline verification only
│   ├── gpu_8gb.json          # Consumer GPU (RTX 3060/4060): batch 32, AMP on
│   └── gpu_16gb_plus.json    # HPC-class GPU (V100/A100): batch 128, AMP on
├── src/
│   ├── dataset.py          # HAM10000Dataset, transforms, lesion-stratified split
│   ├── model.py             # EfficientNet-B0 classifier, checkpoint loading
│   ├── train.py              # Training loop: class-weighted loss, AMP, early stopping
│   ├── evaluate.py           # 7-class + binary malignant/benign metrics
│   └── predict.py            # Single-image CLI inference
└── checkpoints/              # best_model.pth lands here after training (gitignored)
```

## Setup

```bash
git clone https://github.com/motazalqaoud/oncology-diagnosis-cnn
cd oncology-diagnosis-cnn
pip install -r requirements.txt
```

Download HAM10000 from the [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)
(`HAM10000_metadata.csv` + both image archives), unzip so you have:

```
HAM10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

Then verify the layout:

```bash
python data_prep.py --data_dir HAM10000
```

## Training

```bash
python src/train.py --data_dir HAM10000 --config configs/gpu_8gb.json
```

Or without a preset, setting everything manually:

```bash
python src/train.py --data_dir HAM10000 --epochs 30 --batch_size 32
```

Any explicit flag overrides the config file, so `--config configs/cpu.json --epochs 5`
runs the CPU preset's batch size with a shorter run.

### Pick your hardware preset

| Config | Hardware | Batch size | AMP |
|---|---|---|---|
| `configs/cpu.json` | No GPU (pipeline verification only) | 8 | Off |
| `configs/gpu_8gb.json` | RTX 3060 / 4060 | 32 | On |
| `configs/gpu_16gb_plus.json` | V100 / A100 (HPC) | 128 | On |

This trains with class-weighted cross-entropy, mixed precision (`--amp`, on by
default in the GPU presets), `ReduceLROnPlateau` scheduling, and early
stopping on validation loss (default patience: 6 epochs). The best checkpoint
is saved to `checkpoints/best_model.pth`, along with a `training_log.csv` of
per-epoch metrics.

Add `--freeze_backbone` for a fast sanity-check run that only trains the
classification head.

## CUDA setup (GPU training)

If training on an HPC cluster (SLURM) or your own NVIDIA GPU:

**1. Install NVIDIA drivers** (skip on a shared HPC — already provided by the cluster)
```bash
nvidia-smi   # verify: should show your GPU name and driver version
```

**2. Install PyTorch with CUDA**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Verify CUDA is visible to PyTorch**
```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

**On a SLURM cluster**, request a GPU node and load the relevant modules before
the steps above, e.g.:
```bash
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
module load cuda anaconda3
```
(exact module names vary by cluster — check `module avail`).

If CUDA isn't available, `train.py` automatically falls back to CPU — slower,
but functional. Use `configs/cpu.json` in that case.

## Evaluation

```bash
python src/evaluate.py --data_dir HAM10000 --checkpoint checkpoints/best_model.pth
```

Reports, on the held-out test split:
- Full 7-class precision/recall/F1 and confusion matrix
- Macro-average ROC-AUC (one-vs-rest)
- **Binary malignant-vs-benign sensitivity, specificity, and AUC** — the
  clinically actionable numbers — plus the raw false-negative count (missed
  malignant lesions), which is the single most important number to scrutinize
  before considering any downstream use.

## Inference on a single image

```bash
python src/predict.py --image path/to/lesion.jpg --checkpoint checkpoints/best_model.pth
```

## Running the demo locally

```bash
python app.py
```

**If `checkpoints/best_model.pth` doesn't exist yet, the app runs in a
clearly-labeled demo mode**: the full pipeline (image preprocessing,
EfficientNet-B0 forward pass, softmax) executes for real, but the
classification head is untrained, so the displayed percentages aren't
meaningful. Train a model with `src/train.py` and the app will automatically
detect the checkpoint and switch to real predictions on next launch.

## Results

**No checkpoint is bundled in this repo yet** — training requires the 2.7GB
dataset (not included, see Setup) and real compute time. Once you train a
model, run `src/evaluate.py` and paste the output here:

| Region | Dice / Metric | Value |
|---|---|---|
| 7-class accuracy | — | *(run evaluate.py)* |
| Malignant sensitivity | — | *(run evaluate.py)* |
| Malignant specificity | — | *(run evaluate.py)* |
| Macro ROC-AUC | — | *(run evaluate.py)* |

**Fastest path to real numbers:** HAM10000 (2.7GB) is far smaller than the 12K
Kaggle brain tumor set trained in
[Brain-Tumor-Segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation) —
this will train in well under an hour on any free-tier GPU notebook (Kaggle,
Colab), no HPC queue required. Kaggle also hosts a direct HAM10000 mirror you
can attach to a notebook without the Harvard Dataverse download.

For context on what a correctly-trained model of this type should achieve,
published results on this exact dataset using comparable transfer-learning
CNNs report:

| Approach (literature) | Accuracy | Notes |
|---|---|---|
| Inception-V3 transfer learning | ~85% | Alam et al. |
| ResNet50 transfer learning | ~82% | Akter et al. |
| MobileNetV2 fine-tuned | ~95.1% | 7-class, AUC 0.94 |
| Multimodal (image + patient metadata) | ~94.1% | AUC 0.943 |

*(See references below.)* These are reported numbers from published work on
HAM10000, included here to set a realistic expectation range — **not**
results from this specific checkpoint.

## Limitations and disclaimer

**This is a research and portfolio project, not a medical device.** It has
not been validated prospectively, has not been reviewed by a regulatory body,
and must never be used to make or defer an actual diagnosis. HAM10000 also
skews toward lighter Fitzpatrick skin types, a well-documented limitation of
most public dermatology datasets; a model trained only on this data should
not be assumed to generalize equally across all skin tones. See a
dermatologist for any concerning skin lesion.

## Tech stack

| Tool | Purpose |
|---|---|
| `PyTorch` / `torchvision` | Deep learning, EfficientNet-B0 backbone |
| `pandas` | HAM10000 metadata handling |
| `scikit-learn` | Classification metrics, ROC-AUC |
| `Pillow` | Image loading |
| `Gradio` | Interactive demo |

## About the author

**Motaz Alqaoud, PhD**
- PhD in Biomedical Engineering with focus on medical image analysis and deep learning
- Senior AI/ML Engineer specializing in medical imaging, segmentation models, and clinical AI systems
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

## References

- Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset.
  *Scientific Data*, 5, 180161.
- Alam, T.M. et al. Skin lesion classification with Inception-V3 on HAM10000.
- Akter, M. et al. Skin lesion classification with ResNet50 on HAM10000.
- Multimodal (ALBEF) skin lesion classification on HAM10000, medRxiv 2024.

## License

MIT — see [LICENSE](LICENSE).
