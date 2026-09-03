# DACL-CD

**Dual Noise Augmented Consistency Learning for Semi-Supervised Building Change Detection**

Official PyTorch implementation of **DACL-CD**, a semi-supervised learning framework for building change detection (BCD) from bi-temporal remote sensing images. The method integrates **dual-noise augmentation CutMix learning**, a **progressive teacher–student consistency learning framework**, and a **feature difference-weighted spatial attention module (FDSAM)** to effectively exploit unlabeled data under limited annotations.

> Paper: [*Dual Noise Augmented Consistency Learning for Semi-Supervised Building Change Detection*](YOUR_PAPER_URL) (please replace with your paper link)

---

## Table of Contents

- [Highlights](#highlights)
- [Framework Overview](#framework-overview)
- [Main Contributions](#main-contributions)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Highlights

- 🎯 **Dual-Noise Augmentation CutMix Learning (DNACL)** — introduces *Simulated Geometric Offset* (SGO: random translation/rotation/scaling) and *Simulated Radiometric Difference* (SRD: brightness/contrast/HSV adjustment) to generate realistic hybrid bi-temporal samples, alleviating the distribution bias between pseudo-samples and real data.
- 🔁 **Progressive Teacher–Student Consistency Learning** — a three-stage training paradigm (supervised initialization → unsupervised collaboration → consistency constraint) with an EMA-updated teacher, which makes full use of unlabeled data and mitigates overfitting caused by imbalanced change types.
- 🔍 **Feature Difference-weighted Spatial Attention Module (FDSAM)** — explicitly constrains bi-temporal feature discrepancies to highlight subtle and low-contrast change regions while suppressing background noise, improving the localization accuracy of small buildings.
- ⚡ **Lightweight & Efficient** — only **44.87M** parameters, **14.24G** FLOPs, **4.03h** training time and **7.23GB** GPU memory, with state-of-the-art results on three public BCD benchmarks under low label ratios.

---

## Framework Overview

![DACL-CD Framework](./figs/framework.png)

**Figure 1.** Overall architecture of the proposed DACL-CD framework, consisting of a supervised stage and an unsupervised (teacher–student consistency learning) stage.

![DNACL Pipeline](./figs/dnacl.png)

**Figure 2.** Flow chart of the Dual Noise Augmentation CutMix Learning (DNACL).

> 📌 Place `framework.png` and `dnacl.png` in the `figs/` directory (or update the paths to your own figures).

---

## Main Contributions

1. **Dual-Noise Augmentation CutMix Learning (DNACL).** Based on CutMix, we introduce a dual-noise augmentation strategy that simulates real geometric offset (SGO: random translation of ±8 px, rotation of ±15°, scaling of ±10%) and radiometric difference (SRD: brightness/contrast ±20%, HSV hue/saturation/value ±10%) to construct noise-aware mixed bi-temporal samples, reducing the distribution deviation between pseudo-samples and real data and improving robustness to complex noisy environments.

2. **Progressive Teacher–Student Consistency Learning Framework.** We design a collaborative mechanism of "teacher guidance – student exploration" that jointly regularizes dual-temporal feature representations and prediction distributions through staged consistency optimization. The teacher is updated via an Exponential Moving Average (EMA, α = 0.99), producing stable pseudo-labels and alleviating pseudo-label confirmation bias and overfitting under imbalanced change scenarios.

3. **Feature Difference-weighted Spatial Attention Module (FDSAM) & Hierarchical Multi-Scale Fusion.** The FDSAM explicitly constrains bi-temporal feature discrepancies to emphasize subtle and low-contrast change regions while suppressing background responses. Hierarchical multi-scale difference features are progressively aggregated in a pyramid decoder for robust cross-scale interaction and precise spatial localization.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DACL-CD.git
cd DACL-CD

# (Recommended) create a conda environment
conda create -n dacl python=3.9 -y
conda activate dacl

# Install dependencies
pip install -r requirements.txt
```

### Environment

- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- NVIDIA GPU (e.g., RTX 4090) recommended

---

## Data Preparation

We use three public building change detection (BCD) datasets:

| Dataset | Original Images | Resolution | Cropped Pairs (train / val / test) | Split Ratio | Download |
|---|---|---|---|---|---|
| [LEVIR-CD](https://github.com/justchenhao/LEVIR_CD) | 637 × 1024×1024 | 0.5 m | 7120 / 1024 / 2048 | 7:1:2 | [link](https://github.com/justchenhao/LEVIR_CD) |
| [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) | 2 × 32507×15354 | 0.2 m | 6096 / 762 / 762 | 8:1:2 | [link](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) |
| [S2Looking](https://github.com/S2Looking/Dataset) | 5000 × 1024×1024 | 0.5–0.8 m | 17090 / 2423 / 4798 | 7:1:2 | [link](https://github.com/S2Looking/Dataset) |

Crop the original images to **256 × 256** following the paper, then organize the dataset folder as:

```
datasets/
├─ LEVIR-CD/
│  ├─ train/
│  │  ├─ A/          # T1 (pre-temporal) images
│  │  ├─ B/          # T2 (post-temporal) images
│  │  └─ label/      # binary change masks
│  ├─ val/
│  └─ test/
├─ WHU-CD/
└─ S2Looking/
```

---

## Usage

### Training

```bash
# Train on LEVIR-CD with 5% labeled samples
python train.py --dataset LEVIR-CD --label_ratio 0.05

# Train on WHU-CD with 10% labeled samples
python train.py --dataset WHU-CD --label_ratio 0.10
```

**Key hyper-parameters** (defaults follow the paper):

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name: `LEVIR-CD` / `WHU-CD` / `S2Looking` | `LEVIR-CD` |
| `--label_ratio` | Supervised labeled ratio: `0.05 / 0.10 / 0.20 / 0.40` | `0.05` |
| `--epochs` | Total training epochs (30 supervised + 70 unsupervised) | `100` |
| `--batch_size` | Training batch size | `32` |
| `--lr` | AdamW learning rate | `1e-4` |
| `--weight_decay` | Weight decay | `1e-4` |
| `--alpha_ema` | EMA smoothing coefficient for teacher model | `0.99` |
| `--mask_size` | Salient mask size used in DNACL | `64` |
| `--gpu` | GPU id | `0` |

### Testing / Evaluation

```bash
python test.py --dataset LEVIR-CD --checkpoint ./weights/dacl_cd_levir_5pct.pth
```

**Evaluation metrics:** Precision, Recall, F1-Score, IoU, Overall Accuracy (OA).

---

## Experimental Results

### Main Results at Low Label Ratios

| Dataset | Label Ratio | F1 (%) | IoU (%) | OA (%) |
|---|---|---|---|---|
| **LEVIR-CD** | 5% | **87.93** | **78.46** | **98.79** |
| **WHU-CD** | 5% | **89.01** | **80.20** | **99.13** |
| **S2Looking** | 10% | **65.71** | **48.93** | **97.55** |

DACL-CD consistently outperforms state-of-the-art semi-supervised methods (Only-sup, RCR, RCL, FPA, UniMatch, CutMix-CD) across all labeling scenarios, and approaches fully-supervised accuracy with only 5% labeled data. Full results under 10% / 20% / 40% label ratios are reported in the paper.

### Ablation Study (LEVIR-CD, 20% label ratio)

| DNACL | FDSAM | F1 (%) | IoU (%) |
|---|---|---|---|
| ✗ | ✗ | 88.75 | 79.77 |
| ✗ | ✓ | 89.53 | 81.04 |
| ✓ | ✗ | 89.49 | 80.97 |
| ✓ | ✓ | **89.76** | **81.42** |

### Ablation of Each Noise in DNACL (LEVIR-CD, 20% label ratio)

| SGO | SRD | F1 (%) | IoU (%) |
|---|---|---|---|
| ✗ | ✗ | 88.75 | 79.77 |
| ✗ | ✓ | 89.03 | 80.22 |
| ✓ | ✗ | 89.05 | 80.26 |
| ✓ | ✓ | **89.49** | **80.97** |

### Model Efficiency & Complexity

Compared on LEVIR-CD (100 epochs, 256×256 input, batch size 16):

| Method | Params (M) | FLOPs (G) | Training Time (h) | GPU Memory (GB) |
|---|---|---|---|---|
| RCR | 50.69 | 73.23 | 8.08 | 26.30 |
| RCL | 46.85 | 92.59 | 4.70 | 23.47 |
| FPA | 46.85 | 73.23 | 6.25 | 30.28 |
| UniMatch | 40.47 | 16.28 | 7.28 | 25.52 |
| CutMix-CD | 46.85 | 73.23 | 4.32 | 11.68 |
| **DACL-CD** | **44.87** | **14.24** | **4.03** | **7.23** |

DACL-CD requires only 7.23GB GPU memory (38%–76% lower than other methods), ~1/5 of the FLOPs of RCR/FPA, and the shortest training time among all comparative methods.

### Cross-Dataset Generalization (LEVIR → WHU)

Trained on LEVIR-CD and tested on WHU-CD, DACL-CD achieves the best F1 at all label ratios (e.g., **60.45** at 5%, **71.78** at 20%), demonstrating strong generalization across different imaging conditions, building types and scenes.

---

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{li2026daclcd,
  title={Dual Noise Augmented Consistency Learning for Semi-Supervised Building Change Detection},
  author={Li, Zhenxing and Li, Wenzhuo and Zhang, Hongjuan and Liu, Jin and Sun, Kaimin},
  journal={XXX},
  year={2026}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China Projects (NSFC) under Grant **No. 41801344**.

---

## License

This repository is released for **academic research only**. Please contact the authors for commercial use.

---

## Contact

- **Zhenxing Li** — School of Electronic Information, Wuhan University of Science and Technology
- **Wenzhuo Li** (Corresponding author) — liwenzhuo@wust.edu.cn
- **Hongjuan Zhang** (Corresponding author) — hongjuanzhang@whu.edu.cn
