# GUS: Graph Utility Scoring Framework

**GUS (Graph Utility Scoring)** is a decision framework that answers one critical question before you invest weeks building a Graph Neural Network:

> *"Will a GNN actually outperform XGBoost on my fraud detection problem, or am I wasting my time?"*

GUS gives you a data-driven answer in **under an hour**, using only raw dataset characteristics.

---

## The Problem GUS Solves

Teams often spend weeks building complex GNN pipelines, only to discover that XGBoost would have achieved the same (or better) results. GUS prevents this waste by predicting GNN value **before** you build it.

**Without GUS:** Build GNN (2-4 weeks) → Discover XGBoost was sufficient → Wasted effort

**With GUS:** Run GUS screening (1 hour) → Know which model to build → Ship faster

---

## How GUS Works

GUS computes three key metrics from your dataset:

### 1. Feature Predictiveness (FP)
How well can features alone predict fraud? Measured via Random Forest AUC on raw features.

```
High FP (>0.85) → Features are strong → XGBoost may be sufficient
Low FP (<0.70)  → Features are weak  → Graph structure might help
```

### 2. Fraud Homophily (FH)
Do fraudsters connect to other fraudsters? Measured as the fraction of fraud node neighbors that are also fraud.

```
High FH (>0.3) → Fraud clusters in graph → GNN can exploit this
Low FH (<0.1)  → Fraud is scattered     → Graph structure won't help
```

### 3. Effective Graph Utility (EGU)
The key decision metric combining structure quality with improvement headroom:

```
EGU = Graph_Utility × (1 - XGBoost_AUC)
```

Where `Graph_Utility = Adjusted_Homophily × (1 - Feature_Predictiveness)`

### Decision Rule

```
If EGU > 0.01 → Build the GNN (potential improvement exists)
If EGU ≤ 0.01 → Ship XGBoost (GNN won't help much)
```

**Accuracy:** 75% on 12 real-world fraud detection datasets

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost torch torch-geometric
```

### 2. Run GUS Screening

```bash
# Step 1: Compute GUS metrics (homophily, feature predictiveness, graph utility)
python analyze_datasets.py --data_path your_data.csv

# Step 2: Run baselines and get GUS decision
python complete_gnn_experiments.py --datasets your_dataset
```

### 3. Interpret Results

GUS outputs:
- **Feature Predictiveness** score
- **Fraud Homophily** score
- **Effective Graph Utility** score
- **Recommendation:** GNN or XGBoost

---

## Project Structure

```
gus_framework/
├── analyze_datasets.py         # GUS ENTRY POINT - computes all metrics
├── complete_gnn_experiments.py # EGU calculation, decision rule, experiments
├── fdb_experiments.py          # XGBoost baseline generation
│
├── models/
│   └── baselines.py            # XGBoost, GCN, GraphSAGE, GAT for comparison
│
├── data/
│   ├── data_loader.py          # Dataset loading utilities
│   └── graph_builder.py        # Graph construction from tabular data
│
├── utils/
│   └── metrics.py              # AUC, AP, F1, Precision@K, Recall@K
│
├── configs/
│   └── config.py               # Configuration settings
│
├── teaching/
│   ├── gus_framework_explained.html  # Complete GUS explanation
│   └── vp_brief.html                 # Executive summary
│
└── EXPERIMENT_RESULTS.md       # Validation results on 12 datasets
```

---

## GUS Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Feature Predictiveness | RF AUC on features | How much signal is in features alone |
| Fraud Homophily | Mean(fraud neighbor ratio) | Do fraudsters cluster together |
| Adjusted Homophily | (H - H_random) / (1 - H_random) | Homophily corrected for class imbalance |
| Graph Utility | Adj_Homophily × (1 - FP) | Raw graph value potential |
| Effective Graph Utility | GU × (1 - XGB_AUC) | Graph value × improvement headroom |

---

## Validated Results

GUS was validated on 12 real-world fraud detection datasets:

| Dataset | Graph Utility | XGB AUC | GUS Recommendation |
|---------|---------------|---------|-------------------|
| Ecommerce | 0.06 | 0.78 | GNN |
| Vehicle Loan | 0.05 | 0.66 | GNN |
| IP Blocklist | 0.29 | 0.91 | GNN |
| Customs | 0.11 | 0.99+ | XGBoost |
| Yelp | 0.10 | 0.93 | Toss-up |
| Twitter Bots | 0.09 | 0.94 | Toss-up |
| Amazon | 0.12 | 0.98 | XGBoost |
| Elliptic | 0.15 | 0.99+ | XGBoost |
| Credit Card | 0.05 | 0.96 | XGBoost |

**Key Finding:** Effective Graph Utility correlates with GNN improvement (r=0.64, p<0.05)

---

## AUC Prediction Formula

GUS can predict expected baseline AUC before running experiments:

```
Predicted_AUC = 0.611 × Feature_Pred
              - 0.038 × Fraud_Homo
              + 0.076 × Fraud_Rate
              + 0.398

R² = 0.61
```

---

## API Usage

```python
from analyze_datasets import (
    compute_homophily,
    compute_class_homophily,
    compute_feature_predictiveness,
    compute_adjusted_homophily
)

# Load your data
X, y, edge_index = load_your_data()

# Compute GUS metrics
fp = compute_feature_predictiveness(X, y)
fh = compute_class_homophily(edge_index, y, target_class=1)
adj_h = compute_adjusted_homophily(edge_index, y)

# Compute Graph Utility
graph_utility = adj_h * (1 - fp)

# Run XGBoost to get baseline AUC
xgb_auc = run_xgboost_baseline(X, y)

# Compute Effective Graph Utility
egu = graph_utility * (1 - xgb_auc)

# GUS Decision
if egu > 0.01:
    print("Recommend GNN - potential improvement exists")
else:
    print("Recommend XGBoost - GNN won't add much value")
```

---

## Documentation

- **[GUS Framework Explained](teaching/gus_framework_explained.html)** - Full technical deep-dive
- **[VP Brief](teaching/vp_brief.html)** - Executive summary for stakeholders
- **[Experiment Results](EXPERIMENT_RESULTS.md)** - Validation data and statistics

---

## Citation

```bibtex
@inproceedings{gus2026,
  title={GUS: Graph Utility Scoring for Efficient Model Selection in Fraud Detection},
  author={Anonymous},
  booktitle={IJCAI},
  year={2026}
}
```

---

## FAQ

**Q: How long does GUS screening take?**
Under 1 hour for most datasets. Feature predictiveness requires training a Random Forest, which is the slowest step.

**Q: What if GUS says "toss-up"?**
Default to XGBoost - it's faster to deploy and maintain. Only build the GNN if you have engineering capacity and the potential uplift justifies the investment.

**Q: Is GUS specific to fraud detection?**
The core concepts (homophily, feature predictiveness) apply broadly to node classification. The specific thresholds were calibrated on fraud datasets.

**Q: What if my graph structure is different?**
GUS works on any graph. Update `data/graph_builder.py` to construct edges from your entity relationships.

**Q: What GNN should I use if GUS recommends it?**
GUS is model-agnostic. Use any GNN architecture (GCN, GraphSAGE, GAT) appropriate for your problem. The `models/baselines.py` file includes reference implementations.
