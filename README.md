# 🪙 bitcoin-outlier-impact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-red)](https://xgboost.readthedocs.io/)

**Does removing a handful of noisy candles improve Bitcoin price forecasts more than switching to a cutting‑edge model?**  
This repository contains the full code and results of a university research project that answers exactly that.

---

## 🧪 The Experiment

- **Data:** Hourly Bitcoin closing prices during the crash week of **1–7 August 2024** (Binance public API).
- **Outlier Detection:** Z‑Score method (threshold = 2.5).  
  4 extreme candles were flagged – the core of the crash.
- **Models Tested:**
  - 📏 Linear Regression (baseline)
  - 🧠 XGBoost (gradient‑boosted trees)
  - 🤖 MLP Neural Network (scikit‑learn’s `MLPRegressor`)
- **Two Scenarios:**
  1. Train on **all** data (dirty set).
  2. Train on **cleaned** data (outliers removed).
- **Evaluation:** RMSE, MAE, MAPE, and R² on the following **test week** (8–14 Aug 2024).

---

## 📊 Key Findings

| Model | Dirty RMSE | Clean RMSE | RMSE Improvement | R² (Clean) |
|-------|------------|------------|------------------|------------|
| Linear Regression | $339 | $333 | +1.7% | **0.867** |
| XGBoost | $1,042 | $780 | **+25.2%** | 0.268 |
| MLP Neural Net | $1,382 | $1,180 | +14.6% | -0.675 |

💡 **Takeaway:**  
A simple linear regression trained on clean data achieved an R² of **0.867** – explaining 86.7% of the price variance.  
Meanwhile, complex models suffered dramatically when fed dirty data. **Before you chase a fancier model, inspect your data.**

![Model Comparison](output/model_comparison.png)

---

## 🚀 How to Run

**Clone the repo**
   ```bash
   git clone https://github.com/yourusername/bitcoin-outlier-impact.git
   cd bitcoin-outlier-impact
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
Run the analysis

The script will:

    Download the hourly BTC data (or use cached CSV)

    Detect outliers via Z‑Score

    Train and evaluate the three models on dirty & clean sets

    Print a detailed metrics table

    Save the two comparison charts in the output/ folder