# MGDC-Rec: Multi-Granularity Disentangled Contrastive Learning for POI Recommendation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"Multi-Granularity Disentangled Contrastive Learning with Multi-Strategy Enhanced Training for POI Recommendation"**.

## 📋 Overview

MGDC-Rec is a novel Point-of-Interest (POI) recommendation framework that addresses two critical challenges in POI recommendation:
1. **Intent Entanglement**: Disentangles complex user intents across multiple granularities
2. **Data Sparsity**: Employs multi-strategy enhanced training to improve model robustness

### Key Features

- 🎯 **Multi-Granularity Modeling**: Constructs three complementary hypergraphs to capture:
  - Sequential dependencies
  - Global transition patterns
  - High-order collaborative signals

- 🔄 **Disentangled Contrastive Learning**: Maximizes mutual information at both cross-view and user-POI levels

- 💪 **Multi-Strategy Enhanced Training**:
  - Mixed negative sampling (hard, popular, and random negatives)
  - Fast Gradient Method (FGM) based adversarial training
  - Focal Loss for handling class imbalance

- 📊 **Superior Performance**: Significantly outperforms state-of-the-art baselines on real-world datasets

## 🏗️ Architecture

MGDC-Rec consists of three main components:

### 1. Multi-View Hypergraph Construction
- **Multi-View Hypergraph Convolution Network**: Captures user-POI collaborative relationships
- **Directed Hypergraph Convolution Network**: Models POI-to-POI transition patterns
- **Sequence Graph Representation Network**: Encodes sequential dependencies in user trajectories

### 2. Multi-Granularity Disentangled Contrastive Learning
- Cross-view contrastive learning
- User-POI level contrastive learning
- Mixed loss combination: InfoNCE + Focal Loss + Negative Sampling Loss

### 3. Multi-Strategy Enhanced Training
- **Negative Sampling**: Strategic mixture of hard, popular, and random negatives
- **Adversarial Training**: FGM-based perturbations for robustness
- **Adaptive Fusion**: Gate mechanisms to combine multi-view representations

## 📊 Datasets

The model is evaluated on two real-world benchmark datasets:

| Dataset | #Users | #POIs | #Check-ins | Sparsity |
|---------|--------|-------|------------|----------|
| NYC     | 834    | 3,835 | -          | High     |
| TKY     | 2,173  | 7,038 | -          | High     |

**Data Format:**
- `train_poi_zero.txt`: Training trajectories
- `test_poi_zero.txt`: Testing trajectories
- `*_pois_coos_poi_zero.pkl`: POI geographic coordinates

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.3 (for GPU support)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MGDC-Rec.git
cd MGDC-Rec
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The project uses PyTorch with CUDA 11.3. If you have a different CUDA version, please install the corresponding PyTorch version from [pytorch.org](https://pytorch.org/).

### Quick Start

#### Training on NYC Dataset

```bash
python run.py \
    --dataset NYC \
    --num_epochs 30 \
    --batch_size 200 \
    --lr 1e-3 \
    --emb_dim 128 \
    --dropout 0.4 \
    --num_mv_layers 3 \
    --num_di_layers 3 \
    --lambda_cl 0.05 \
    --temperature 0.1 \
    --focal_alpha 0.5 \
    --focal_gamma 1.5 \
    --num_neg_samples 8 \
    --adv_epsilon 1.0
```

#### Training on TKY Dataset

```bash
python run.py \
    --dataset TKY \
    --num_epochs 30 \
    --batch_size 200 \
    --lr 1e-3 \
    --emb_dim 128
```

## ⚙️ Configuration

### Model Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--emb_dim` | Embedding dimension | 128 |
| `--dropout` | Dropout rate | 0.4 |
| `--num_mv_layers` | Number of multi-view hypergraph layers | 3 |
| `--num_di_layers` | Number of directed hypergraph layers | 3 |
| `--lambda_cl` | Weight of contrastive loss | 0.05 |
| `--temperature` | Temperature for contrastive learning | 0.1 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_epochs` | Number of training epochs | 30 |
| `--batch_size` | Input batch size | 200 |
| `--lr` | Learning rate | 1e-3 |
| `--decay` | Weight decay (L2 regularization) | 1e-3 |

### Loss Function Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--focal_alpha` | Focal Loss alpha parameter | 0.5 |
| `--focal_gamma` | Focal Loss gamma parameter | 1.5 |
| `--num_neg_samples` | Number of negative samples | 8 |
| `--adv_epsilon` | Adversarial perturbation magnitude | 1.0 |

## 📈 Results

The model is evaluated using the following metrics:
- **Recall@K**: Hit rate in top-K recommendations (K = 1, 5, 10, 20)
- **NDCG@K**: Normalized Discounted Cumulative Gain (K = 1, 5, 10, 20)

Training logs and models are automatically saved in the `logs/` directory with timestamps.

### Performance Highlights

MGDC-Rec significantly outperforms state-of-the-art baselines on both datasets, with particular strength in:
- ✅ Handling sparse user-POI interactions
- ✅ Capturing multi-granularity user preferences
- ✅ Robust performance under data incompleteness
- ✅ Interpretable disentangled representations

## 📁 Project Structure

```
MGDC-Rec/
├── dataset.py              # Dataset loading and preprocessing
├── model.py                # MGDC-Rec model implementation
├── metrics.py              # Evaluation metrics (Recall, NDCG)
├── utils.py                # Utility functions
├── run.py                  # Training and evaluation script
├── requirements.txt        # Python dependencies
├── datasets/
│   ├── NYC/               # NYC dataset files
│   └── TKY/               # Tokyo dataset files
├── logs/                  # Training logs and saved models
└── README.md              # This file
```

## 🔍 Model Components

### Core Modules

1. **ContrastiveLearningModule** (`model.py`): Implements InfoNCE loss with projection heads
2. **NegativeSamplingLoss** (`model.py`): Mixed negative sampling strategy
3. **FGM** (`model.py`): Fast Gradient Method for adversarial training
4. **MultiViewHyperConvNetwork** (`model.py`): Multi-view hypergraph convolution
5. **DirectedHyperConvNetwork** (`model.py`): Directed POI transition modeling
6. **SeqGraphRepNetwork** (`model.py`): Sequential trajectory encoding

### Loss Functions

The total loss is computed as:

```
Total Loss = Main Loss + λ_cl × Contrastive Loss + Adversarial Loss
```

Where:
- **Main Loss** = (1 - w) × Focal Loss + w × Ranking Loss
- **Contrastive Loss** = Multi-View Loss + User-POI Loss + Cross-View Loss
- **Adversarial Loss** = Loss computed on perturbed embeddings

## 🛠️ Advanced Usage

### Custom Dataset

To use your own dataset:

1. Prepare data files in the format:
   - Training/testing trajectories: pickle files containing `[sessions_dict, labels_dict]`
   - POI coordinates: pickle file with `{poi_id: (lat, lon)}`

2. Update dataset parameters in `run.py`:
```python
if args.dataset == "YOUR_DATASET":
    NUM_USERS = your_num_users
    NUM_POIS = your_num_pois
    PADDING_IDX = NUM_POIS
```

3. Run training:
```bash
python run.py --dataset YOUR_DATASET
```

### Monitoring Training

Logs are saved in `logs/YYYYMMDD_HHMMSS/log_training.txt` and include:
- Training and testing loss per epoch
- Recall@K and NDCG@K metrics
- Model hyperparameters
- Best model checkpoints

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liang2025mgdc,
  title={Multi-Granularity Disentangled Contrastive Learning with Multi-Strategy Enhanced Training for POI Recommendation},
  author={Liang, Jiarui and Zuo, Jiankai and Zhang, Yaying},
  journal={},
  year={2025},
  organization={Tongji University}
}
```

## 👥 Authors

- **Jiarui Liang** - Tongji University
- **Jiankai Zuo** - Tongji University
- **Yaying Zhang** - Tongji University (Corresponding Author)

*Key Laboratory of Embedded System and Service Computing of Ministry of Education, Tongji University, Shanghai, China*

## 📧 Contact

For questions or collaboration, please contact:
- Jiarui Liang: [email]
- Yaying Zhang: [email] (Corresponding Author)

## 🙏 Acknowledgments

This work builds upon recent advances in:
- Graph Neural Networks for recommendation
- Contrastive learning methods
- Hypergraph representation learning
- POI recommendation systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [Hypergraph Contrastive Collaborative Filtering (HCCF)](https://dl.acm.org/doi/10.1145/3477495.3531735)
- [DisenPOI: Disentangling Sequential and Geographical Influence](https://dl.acm.org/doi/10.1145/3539597.3570409)
- [Multi-view Spatial-Temporal Enhanced Hypergraph Network](https://link.springer.com/chapter/10.1007/978-3-031-30672-3_16)

## 🐛 Known Issues

- GPU memory usage can be high for large batch sizes (adjust `--batch_size` accordingly)
- CUDA 11.3 specific builds for PyTorch geometric extensions

## 🔄 Updates

- **2025-01**: Initial release

---

⭐ If you find this project helpful, please consider giving it a star!

