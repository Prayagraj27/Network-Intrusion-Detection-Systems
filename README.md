# Network-Intrusion-Detection-Systems
This is a comparative study between the capabilities of ML and RL based Intrusion Detection Systems (IDS).

# Adaptive Network Intrusion Detection via Deep Reinforcement Learning

> *Can Reinforcement Learning Overcome the Static-Training Bottleneck in Network Intrusion Detection?*  
> *An Evaluation on CIC IDS 2017*

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIC%20IDS%202017-4CAF50?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blueviolet?style=flat-square)

---

## Overview

Traditional supervised intrusion detection systems are trained once on a static snapshot of traffic and then deployed frozen — unable to adapt as attack patterns evolve. This project investigates whether a **Deep Q-Network (DQN) agent**, trained *online* on a live traffic stream, can overcome this fundamental limitation.

Using the **CIC IDS 2017** benchmark dataset (78 network flow features, 8 attack classes), I compare six classical ML baselines against a custom **Double Dueling DQN with Prioritised Experience Replay** under a strict streaming evaluation protocol: ML models see 30% of the data for batch training and are then locked; the DQN agent adapts sample-by-sample across the remaining 70%.

---

## Repository Structure

```
.
├── 01_data_preparation.ipynb      # Dataset download, cleaning, EDA, feature engineering
├── 02_ml_baselines.ipynb          # Batch training + streaming evaluation of 6 ML models
├── 03_dqn_agent.ipynb             # Dueling DDQN agent design, training & online evaluation
├── 04_comparative_analysis.ipynb  # Publication-quality figures & master metrics table
├── data/                          # Raw CIC IDS 2017 CSVs (see setup instructions)
├── processed/                     # Preprocessed numpy arrays & fitted scaler (auto-generated)
├── results/                       # Saved model weights, metrics JSON, prediction arrays
└── README.md
```

---

## Methodology

### Evaluation Protocol

The experimental design deliberately mirrors a real deployment scenario:

```
Full Dataset (CIC IDS 2017)
        │
        ├─── 30% ──► Batch Training Set  ──► ML Baselines (frozen after training)
        │
        └─── 70% ──► Streaming Test Set  ──► All models evaluated one sample at a time
                                             DQN continues to update online
```

Temporal ordering is **preserved** (no shuffle) to expose models to realistic concept drift as new attack patterns emerge later in the stream.

### Models Compared

| Model | Type | Adapts Online? |
|---|---|:---:|
| Random Forest | Ensemble | ✗ |
| Gradient Boosting | Ensemble | ✗ |
| Decision Tree | Tree | ✗ |
| Logistic Regression | Linear | ✗ |
| k-NN (k=5) | Instance-based | ✗ |
| Naïve Bayes | Probabilistic | ✗ |
| **Dueling DDQN** | **Deep RL** | **✓** |

### DQN Architecture

The agent frames intrusion detection as a **Markov Decision Process**:

| Component | Definition |
|---|---|
| **State** s_t | Normalised 78-dim network flow feature vector |
| **Action** a_t | `{0 = Benign, 1 = Attack}` |
| **Reward** r_t | `+1.0` correct detection / `+0.8` correct benign / `-1.0` false negative / `-0.5` false positive |

The network implements a **Dueling architecture** with separate Value `V(s)` and Advantage `A(s,a)` streams, combined as:

```
Q(s,a) = V(s) + A(s,a) − mean_a′[A(s,a′)]
```

Key design choices:
- **Double DQN** — decouples action selection from evaluation to reduce overestimation bias
- **Prioritised Experience Replay (PER)** — SumTree for O(log n) proportional sampling; rarer, high-error transitions are replayed more frequently
- **Dual-buffer replay** — maintains a balanced benign/attack buffer to counter class imbalance without undersampling
- **Warm-start** — replay buffer is seeded with 5,000 transitions from the training split to avoid cold-start random exploration

```
Input (78) → FC(256) → LayerNorm → ELU
           → FC(128) → LayerNorm → ELU
           → FC(64)  → LayerNorm → ELU
                           │
              ┌────────────┴─────────────┐
          Value stream            Advantage stream
          FC(32) → V(s)           FC(32) → A(s, a)
                           │
                      Q(s, a)  [combined]
```

---

## Dataset

**Canadian Institute for Cybersecurity — Intrusion Detection System 2017 (CIC IDS 2017)**

- **Features:** 78 network flow statistics (packet lengths, flow duration, inter-arrival times, flag counts, etc.)
- **Labels:** BENIGN + 7 attack families — DoS, DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet

**Download:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

Place all `*_TrafficFlowMeter.csv` files in `./data/` before running Notebook 1.

> **No dataset?** Notebook 1 auto-generates a synthetic surrogate (~250,000 samples) that preserves the approximate class distribution and feature dimensionality of the real dataset, so all four notebooks execute end-to-end without the raw CSVs.

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install torch numpy pandas scikit-learn xgboost imbalanced-learn \
            matplotlib seaborn tqdm joblib
```

### 2. Run notebooks in order

```
01_data_preparation.ipynb   →  cleans data, saves processed/ artefacts
02_ml_baselines.ipynb       →  trains & evaluates ML models, saves results/
03_dqn_agent.ipynb          →  trains DQN online, saves agent weights & metrics
04_comparative_analysis.ipynb → generates all comparison figures & tables
```

Each notebook is self-contained and loads outputs from the previous one via the `processed/` and `results/` directories.

---

## Results

### Key Metrics (Streaming Evaluation — 70% holdout)

| Model | Accuracy | F1-Score | Recall (DR) | AUC-ROC | FPR |
|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| k-NN | — | — | — | — | — |
| Naïve Bayes | — | — | — | — | — |
| **Dueling DDQN** | — | — | — | — | — |

> *Run Notebook 4 to populate this table with your results.*

### Figures Generated

- **Figure A** — Head-to-head metric comparison (grouped bar chart)
- **Figure B** — Rolling accuracy convergence over the stream
- **Figure C** — Concept drift analysis (windowed F1 over time)
- **Figure D** — FPR vs Detection Rate scatter in ROC space
- **Figure E** — DQN learning curve vs best static ML baseline
- **Figure F** — Radar / spider chart (5-metric profile per model)
- **Figure G** — DQN reward evolution & training loss

---

## Technical Highlights

- **Temporal train/test split** — no data leakage; stream preserves chronological order
- **Class imbalance handling** — balanced class weights for ML models; dual-buffer replay + shaped rewards for DQN
- **Concept drift visualisation** — windowed F1 scores reveal degradation in static models over time
- **Prioritised replay with SumTree** — O(log n) sampling vs O(n) naive alternatives
- **Soft target updates** — Polyak averaging (τ = 0.005) for stable Q-value convergence
- **ε-greedy exploration schedule** — linear decay over 50,000 steps from 1.0 → 0.1

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | DQN neural network & training |
| `scikit-learn` | ML baseline models & metrics |
| `xgboost` | Gradient boosting baseline |
| `imbalanced-learn` | SMOTE / undersampling utilities |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `tqdm`, `joblib` | Progress bars & model persistence |

Python 3.9+ recommended.

---

## Limitations & Future Work

- The synthetic surrogate dataset, while dimensionally accurate, cannot replicate the full statistical complexity of real CIC IDS 2017 traffic — results should be validated on the actual dataset.
- The DQN requires access to the true label immediately after each classification (supervised reward signal). A purely unsupervised or semi-supervised reward formulation would be more realistic.
- Potential extensions: PPO / SAC for continuous action spaces; multi-class detection head; federated RL across distributed sensors; concept drift detection with automatic replay buffer purging.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---
