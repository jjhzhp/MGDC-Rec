# MGDC-Rec

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Code](https://img.shields.io/badge/Code-python-purple)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

MGDC-Rec: Multi-Granularity Disentangled Contrastive Learning with Multi-Strategy Enhanced Training for POI Recommendation

-  ***Multi-Granularity Modeling***: Constructs three complementary hypergraphs to capture sequential dependencies, global transition patterns, and high-order collaborative signals.
-  ***Disentangled Contrastive Learning***: Maximizes mutual information at both cross-view and user-POI levels to disentangle complex user intents.
-  ***Multi-Strategy Enhanced Training***: Employs mixed negative sampling (hard, popular, and random negatives), FGM-based adversarial training, and Focal Loss to improve model robustness and handle data sparsity.

## Methodology
To address intent entanglement and data sparsity in POI recommendation, we propose a multi-granularity disentangled contrastive learning framework. The model consists of three main components:

<p align="center">
<img align="middle" src="Figures/fig_model.png" width="800"/>
</p>
<p align = "center">
<b>Figure 1. The overall framework of the proposed MGDC-Rec model.</b>
</p>

1. **Multi-View Hypergraph Construction**: This module constructs three hypergraphs to model user preferences from different perspectives:
    - **Sequence Graph**: Captures sequential dependencies in user trajectories.
    - **Directed Global Transition Graph**: Models POI-to-POI transition patterns globally.
    - **User-POI Collaborative Graph**: Captures high-order collaborative signals.

2. **Multi-Granularity Disentangled Contrastive Learning**: This module applies contrastive learning at both cross-view and user-POI levels to learn disentangled representations of user intents.

3. **Multi-Strategy Enhanced Training**: We integrate mixed negative sampling strategy (combining hard, popular, and random negatives), adversarial training (using Fast Gradient Method), and Focal Loss (to address class imbalance) to significantly enhance model robustness and performance.

<p align="center">
<img align="middle" src="Figures/fig_CL.png" width="600"/>
</p>
<p align = "center">
<b>Figure 2. Illustration of the Multi-Granularity Disentangled Contrastive Learning module.</b>
</p>

<p align="center">
<img align="middle" src="Figures/fig_Enh.png" width="600"/>
</p>
<p align = "center">
<b>Figure 3. Illustration of the Multi-Strategy Enhanced Training.</b>
</p>

## Requirements
The code has been tested running under Python 3.8.

The required packages are as follows: 
- Python >= 3.8
- torch == 1.12.0
- torch_geometric == 2.0.4
- pandas == 2.3.3
- numpy == 1.26.4
- pyyaml == 6.0.3

## Data
This folder (datasets) contains 2 datasets, including

(1) **NYC** (New York City in USA); 

(2) **TKY** (Tokyo in Japan).

| Dataset | #Users | #POIs | #Check-ins | #Sessions |
|---------|--------|-------|------------|----------|
| NYC     | 834    | 3,835 | 45,599     | 8,841    |
| TKY     | 2,173  | 7,038 | 306,778    | 41,307   |

The data format includes:
- `train_poi_zero.txt`: Training trajectories
- `test_poi_zero.txt`: Testing trajectories
- `*_pois_coos_poi_zero.pkl`: POI geographic coordinates

## Running
You can use the NYC dataset as an example to run it as:

```shell
nohup python run.py --dataset NYC --num_epochs 30 --batch_size 200 > NYC.log 2>&1 &
```

For TKY dataset:
```shell
nohup python run.py --dataset TKY > TKY.log 2>&1 &
```

## Result
For a detailed analysis of the results, please refer to the paper. We evaluate our proposed MGDC-Rec model on NYC and TKY datasets using **Recall@K** and **NDCG@K** (K = 1, 5, 10, 20).

The model demonstrates superior performance in:
- Handling sparse user-POI interactions
- Capturing multi-granularity user preferences
- Robust performance under data incompleteness

**We appreciate the efforts of these scholars and their excellent work!**
