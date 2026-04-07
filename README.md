# Meta Guidance: Incorporating Inductive Biases into Deep Time Series Imputers

> A lightweight, model-agnostic module that incorporates **Non-Stationary** and **Periodic** inductive biases into deep time series imputation models, achieving an average **27.39%** error reduction across nine benchmark datasets.

## Authors

**Jiacheng You**<sup>1</sup>, **Xinyang Chen**<sup>1</sup>, **Yu Sun**<sup>2</sup>, **Weili Guan**<sup>3</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)  
<sup>2</sup> College of Computer Science, DISSec, Nankai University  
<sup>3</sup> School of Information Science and Technology, Harbin Institute of Technology (Shenzhen)  
\* Corresponding author

## Links

- **Paper**: NIPS 2025
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/NIPS2025-Meta-Guidance)
- **Base Framework**: [Time-Series-Library (THUML)](https://github.com/thuml/Time-Series-Library)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Demo / Visualization](#demo--visualization)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [04/2026] Initial release

---

## Introduction

本项目是论文 **Meta Guidance: Incorporating Inductive Biases into Deep Time Series Imputers**（NIPS 2025）的官方实现。

时间序列插补是数据分析和机器学习中的关键任务。现有的深度插补方法通常采用端到端学习隐式推断时间模式，忽略了显式融入与时间序列特征对齐的**领域先验知识（inductive biases）**的巨大潜力。本项目利用时间序列的**非平稳性（non-stationarity）**和**周期性（periodicity）**两种特性，提出了两种领域特定的先验引导机制：**非平稳引导（Non-Stationary Guidance, NSG）**和**周期引导（Periodic Guidance, PG）**，并通过**元引导（MetaGuidance, MG）**自适应地融合两者。在九个基准数据集上，将 MetaGuidance 注入现有深度插补架构，平均实现了 **27.39%** 的插补误差降低。

### Core Idea

We present **MetaGuidance (MG)**, a lightweight and model-agnostic guidance mechanism for **time series imputation**. By exploiting the non-stationarity and periodicity properties of time series data, two domain-specific inductive biases are designed:

- **Non-Stationary Guidance (NSG)**: Operationalizes the proximity principle to address highly non-stationary series by emphasizing temporal neighbors. A channel-wise scaling scalar $\gamma^{NS}_c$ is learned via MLP to dynamically adjust the guidance strength.
- **Periodic Guidance (PG)**: Exploits periodicity patterns by extracting the top-$k$ dominant frequencies via FFT and assigning learnable weights to values at corresponding periodic offsets. A period-specific scaling scalar $\gamma^{Per}_{l,c}$ is learned via MLP for each significant frequency.
- **MetaGuidance Fusion**: MG adaptively learns a data-driven weight $\lambda_c \in [0,1]$ (via MLP with Sigmoid activation) to fuse NSG and PG: $g^{Meta}_{:,c} = \lambda_c \cdot g^{NS}_{:,c} + (1 - \lambda_c) \cdot g^{PG}_{:,c}$.
- **Model-Agnostic Design**: MG can be seamlessly injected into diverse deep imputation architectures (Transformer, TimesNet, iTransformer, PatchTST, Nonstationary Transformer, etc.) by modifying the normalization and embedding modules, without changing the core model architectures.

This repository provides:
  - Training and evaluation code for imputation tasks
  - Multiple backbone models enhanced with MetaGuidance
  - Experiment scripts for various benchmark datasets
  - Pre-trained checkpoints

---

## Highlights

- Proposes two imputation-oriented inductive biases: **Non-Stationary Guidance (NSG)** and **Periodic Guidance (PG)**
- Introduces **MetaGuidance (MG)**, a lightweight, model-agnostic module that adaptively fuses NSG and PG via data-driven weights
- Achieves an average **27.39% error reduction** across **nine real-world benchmark datasets**
- Can be seamlessly integrated into diverse deep imputation architectures (Transformer, TimesNet, iTransformer, etc.)
- Provides comprehensive experiment scripts and evaluation metrics (MAE, MSE)

---

## Method / Framework

### Framework Overview

MetaGuidance consists of three core components:

1. **Non-Stationary Guidance (NSG)**: For each missing value at position $(t, c)$, NSG assigns guidance weights to observed values within a neighborhood range $[-r, r]$, weighted by a Gaussian distribution $\psi(i)$ scaled by a learnable channel-wise multiplier $\gamma^{NS}_c$ (learned via MLP). This captures local temporal continuity via the proximity principle.

2. **Periodic Guidance (PG)**: Extracts the top-$k$ dominant frequencies from each channel via FFT, converts them to period lengths $p_{l,c}$, and assigns guidance weights to observed values at periodic offsets ($i \cdot p_{l,c} + t$). Each period is weighted by a learnable scaling scalar $\gamma^{Per}_{l,c}$ (learned via MLP) and an amplitude-based significance score.

3. **MetaGuidance Fusion**: Learns a data-adaptive weight $\lambda_c = \text{MLP}(x', g^{NS}, g^{PG})$ with Sigmoid activation ($\lambda_c \in [0,1]$) to fuse NSG and PG per channel: $g^{Meta} = \lambda_c \cdot g^{NS} + (1 - \lambda_c) \cdot g^{PG}$.

MetaGuidance is injected into the backbone model through two integration points:
- **Weighted Normalization**: Replaces standard RevIN by computing weighted mean and variance using $g^{Meta}$ as guidance weights
- **Guidance Embedding**: A dedicated embedding branch $E_{Meta} = \text{Embedding}(g^{Meta} \odot X)$ concatenated with the standard input embedding

**Figure 1.** Overall framework of injecting Meta Guidance into the Transformer. Compared to the vanilla Transformer, it includes additional components: the Guidance Generator, the Guidance Embedding module, and weighted Normalization / De-normalization modules.

---

## Project Structure

```text
.
├── checkpoints/                # Pre-trained model weights
├── data_provider/               # Data loading and preprocessing
│   └── data_factory_my.py       # Custom data provider for imputation
├── dataset/                     # Datasets (ETT, Weather, Electricity, Traffic, etc.)
│   ├── ETT-small/               # ETT datasets
│   ├── weather/
│   ├── electricity/
│   ├── traffic/
│   ├── HD/
│   ├── TCPC/
│   └── make_miss_csv.py         # Script to create data with missing values
├── exp/                         # Experiment definitions
│   ├── exp_imputation_my.py     # Custom imputation experiment (MetaGuidance)
│   └── exp_basic.py             # Base experiment class
├── layers/                      # Neural network layers
│   ├── MetaGuidance.py          # Core: GuidanceGenerator implementation
│   ├── Invertible_weight_pro.py # Core: Weighted RevIN implementation
│   ├── Embed_my.py              # Custom embeddings with weight-aware embedding
│   ├── Transformer_EncDec.py    # Encoder/Decoder for Transformer
│   └── SelfAttention_Family.py  # Attention mechanisms
├── models/                      # Model implementations
│   ├── Transformer_my.py        # Transformer + MetaGuidance (no interpolation)
│   ├── TimesNet_my.py           # TimesNet + RevIN_weight (with interpolation)
│   ├── iTransformer_my.py       # iTransformer + MetaGuidance (no interpolation)
│   ├── PatchTST_my.py           # PatchTST + RevIN_weight (with interpolation)
│   ├── Nonstationary_Transformer_my.py  # Nonstationary Transformer + RevIN_weight
│   └── *.py                     # Other baseline models
├── scripts/                     # Experiment scripts
│   ├── imputation/              # Imputation experiment scripts
│   ├── long_term_forecast/      # Long-term forecasting scripts
│   ├── short_term_forecast/     # Short-term forecasting scripts
│   ├── anomaly_detection/       # Anomaly detection scripts
│   └── classification/          # Classification scripts
├── results/                     # Experiment result files (.npy)
├── test_results/                # Visualization results (.pdf)
├── pic/                         # Images
│   └── dataset.png
├── utils/                       # Utility functions
├── myrun.py                     # Main entry point (MetaGuidance variant)
├── run.py                       # Main entry point (baseline variant)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: This project requires PyTorch >= 1.7.1. Please install the appropriate PyTorch version for your CUDA environment from [PyTorch official website](https://pytorch.org/).

---

## Checkpoints / Models

Pre-trained checkpoints are provided in the `checkpoints/` directory (32 `.pth` files).

To use a checkpoint for testing:

```bash
python myrun.py \
  --task_name imputation \
  --is_training 0 \
  --model myTransformer \
  ... (other args)
```

---

## Dataset / Benchmark

This project evaluates on the following benchmark datasets:

| Dataset | Variables | Description |
|---|---|---|
| ETTh1 | 7 | Electricity Transformer Temperature (hourly) |
| ETTh2 | 7 | Electricity Transformer Temperature (hourly) |
| ETTm1 | 7 | Electricity Transformer Temperature (15-min) |
| ETTm2 | 7 | Electricity Transformer Temperature (15-min) |
| Weather | 21 | Weather conditions |
| Electricity | 321 | Electricity consumption |
| Traffic | 862 | Highway traffic flow |
| TCPC | 8 | Private dataset |
| HD | 5 | Private dataset |

Data should be placed in the `dataset/` directory. For datasets with missing values, use the `dataset/make_miss_csv.py` script to generate corrupted versions.

> Public datasets (ETT, Weather, Electricity, Traffic) can be obtained from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository.

---

## Usage

### Training (Imputation with MetaGuidance)

Use `myrun.py` for the MetaGuidance variant (no interpolation):

```bash
python myrun.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_norev \
  --model myTransformer \
  --data MyData \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 \
  --mask_rate 0.1 \
  --seed 1
```

Or use provided shell scripts:

```bash
bash ./scripts/imputation/exe.sh
```

### Training (Baseline without MetaGuidance)

Use `run.py` for the baseline variant:

```bash
python run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model Transformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 128 \
  --mask_rate 0.25
```

### Testing

```bash
python myrun.py \
  --task_name imputation \
  --is_training 0 \
  ... (same args as training)
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--task_name` | Task type: `imputation`, `long_term_forecast`, etc. | Required |
| `--model` | Model name: `myTransformer`, `myTimesNet`, `iTransformer_my`, `NS_Transformer_my`, `PatchTST_my`, etc. | Required |
| `--mask_rate` | Missing ratio for imputation | 0.25 |
| `--k` | Number of top periodic components | 3 |
| `--r` | Range of temporal neighbors for guidance | 3 |
| `--weight_rev` | Whether to apply weighted RevIN | True |
| `--seed` | Random seed | 2 |

### Long-term Forecasting

```bash
bash ./scripts/long_term_forecast/ETT_script/<model>_ETTh1.sh
```

---

## Demo / Visualization

Visualization results (PDF) are saved to `test_results/` during testing.

### Example Results (Imputation)

The following table shows imputation results (MAE / MSE) on ETTh1 at 10% mask rate:

| Model | MAE | MSE |
|---|---:|---:|
| Transformer + MetaGuidance | 0.5718 | 1.1294 |
| TimesNet + MetaGuidance | 0.5415 | 0.9432 |

Results are also recorded in `result_imputation.txt`.

---

## TODO

- [ ] Add framework figure to `pic/`
- [ ] Upload checkpoints to Hugging Face
- [ ] Release project page

---

## Citation

If you find this repo useful, please cite our paper:

```bibtex
@article{you2025meta,
  title={Meta Guidance: Incorporating Inductive Biases into Deep Time Series Imputers},
  author={Jiacheng You and Xinyang Chen and Yu Sun and Weili Guan and Liqiang Nie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

Base framework citation:

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

---

## Acknowledgement

- Thanks to the open-source [Time-Series-Library (THUML/Tsinghua)](https://github.com/thuml/Time-Series-Library) for providing the base framework.
- Thanks to the open-source community for providing useful baselines and tools.

---

## License

This project is released under the MIT License.
