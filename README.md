<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Robust%20Regression%20Engine&fontSize=48&fontColor=ffffff&fontAlignY=35&desc=Advanced%20ML%20Pipeline%20%7C%20House%20Price%20Prediction&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<!-- BADGES ROW 1 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.8-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/Status-✅%20Completed-2ea44f?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20R²-0.9274-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20Model-Random%20Forest-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Records-3%2C800%20Properties-blue?style=for-the-badge"/>
</p>

<!-- TAGLINE -->
<br/>
<i>🏠 A production-grade ML regression pipeline — built to beat overfitting, not just pass a notebook check.</i>
<br/><br/>

<!-- SOCIAL LINKS -->
<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram"/>
</a>

<br/><br/>

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 01 | [🎯 Project Overview](#-project-overview) |
| 02 | [📊 Dataset Statistics](#-dataset-statistics) |
| 03 | [🗂️ Dataset Schema](#️-dataset-schema) |
| 04 | [⚙️ ML Pipeline](#️-ml-pipeline) |
| 05 | [🧠 Theory Concepts](#-theory-concepts) |
| 06 | [🤖 Models Implemented](#-models-implemented) |
| 07 | [🏆 Performance Results](#-performance-results) |
| 08 | [📈 Visualizations](#-visualizations-10-charts) |
| 09 | [📁 Project Structure](#-project-structure) |
| 10 | [📋 Summary Report](#-summary-report) |
| 11 | [👤 Author & Contact](#-author--contact) |
| 12 | [📜 License](#-license) |

---

## 🎯 Project Overview

> You are working as a **Machine Learning Engineer** at a real estate analytics company.
> The existing basic regression model suffers from **overfitting and unstable predictions** across different datasets.
> Your task is to build a **robust regression pipeline** that generalizes well on unseen data.

### What this project does:
- ✅ Applies **regularization techniques** (Ridge L2, Lasso L1)
- ✅ Uses **proper cross-validation strategies** (K-Fold, Stratified, TimeSeries, LOOCV)
- ✅ Compares **linear and non-linear regression models**
- ✅ Selects the **best-performing model** based on validation performance

| Field | Details |
|-------|---------|
| 🎓 Exam Type | Practical · 6 Hours |
| 🏢 Domain | Real Estate Analytics |
| 🎯 ML Task | Supervised Regression |
| 📦 Target | `house_price_inr` |
| 🏆 Best Model | Random Forest (R² = 0.9274) |

---

## 📊 Dataset Statistics

<div align="center">

| 🏠 Properties | 🔢 Features | 💰 Avg Price | 📐 Avg Area |
|:---:|:---:|:---:|:---:|
| **3,800** | **11** | **₹20.7M** | **1,717 sqft** |

| 📅 Date Range | 🌲 RF Trees | 🏆 Best R² | 🎯 Test Set |
|:---:|:---:|:---:|:---:|
| **2013–2020** | **100** | **0.9274** | **760 rows** |

</div>

---

## 🗂️ Dataset Schema

| # | Column | Type | Description | Notes |
|---|--------|------|-------------|-------|
| 1 | `property_id` | Int | Unique identifier | Primary Key |
| 2 | `sale_date` | Date | Sale transaction date | → year, month extracted |
| 3 | `area_sqft` | Float | Property size (sq.ft) | ⭐ Top feature |
| 4 | `bedrooms` | Int | Number of bedrooms | — |
| 5 | `bathrooms` | Int | Number of bathrooms | — |
| 6 | `location_score` | Float | Location quality (1–10) | ⭐ Top feature |
| 7 | `property_age` | Int | Age in years | — |
| 8 | `distance_city_km` | Float | Distance to city (km) | 📉 Negative driver |
| 9 | `near_school` | Binary | 1 = near school | — |
| 10 | `near_metro` | Binary | 1 = near metro | — |
| 11 | `crime_rate_index` | Float | Local crime rate | 📉 Negative driver |
| 12 | `house_price_inr` | Float | House price in INR | 🎯 **TARGET** |

---

## ⚙️ ML Pipeline

```
📥 Load Data  →  🧹 Preprocess  →  📐 Ridge/Lasso  →  🔁 Cross-Val  →  🌲 Trees  →  ⚡ SVR  →  📊 Compare  →  ✅ Report
  (Part B)        (Part B)           (Part C)           (Part D)        (Part E)    (Part F)   (Part G)      (Part H)
```

### Steps at a Glance:

```python
# Part B — Data Preparation
df = pd.read_csv('house_price.csv')
df['sale_year'] = pd.to_datetime(df['sale_date']).dt.year
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_sc = StandardScaler().fit_transform(X_train)

# Part C — Regularization
ridge = Ridge(alpha=10).fit(X_train_sc, y_train)       # L2
lasso = Lasso(alpha=10000).fit(X_train_sc, y_train)    # L1

# Part D — Cross-Validation
kf_scores = cross_val_score(Ridge(alpha=10), X_sc, y, cv=KFold(5), scoring='r2')

# Part E — Tree Models
rf = RandomForestRegressor(n_estimators=100, max_depth=10).fit(X_train, y_train)

# Part F — SVR
svr = SVR(kernel='rbf', C=100, gamma=0.01).fit(X_train_sc, y_train)
```

---

## 🧠 Theory Concepts

<details>
<summary><b>Q1. What is Regularization? Why is it needed?</b></summary>
<br>

Regularization adds a **penalty term** to the loss function to discourage overly complex models. It prevents **overfitting** by shrinking model coefficients toward zero, improving generalization on unseen data.

Without regularization → model memorizes training data (high variance, low bias) → fails on new inputs.

</details>

<details>
<summary><b>Q2. Ridge (L2) vs Lasso (L1) Regression</b></summary>
<br>

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| **Penalty** | λΣβ² (squared) | λΣ\|β\| (absolute) |
| **Effect** | Shrinks all coefs toward 0 | Can zero out coefs completely |
| **Feature Selection** | ❌ No | ✅ Yes (automatic) |
| **Geometry** | Circular constraint | Diamond → sparse solutions |
| **Best for** | Multicollinearity | High-dim, many irrelevant features |

</details>

<details>
<summary><b>Q3. Cross-Validation Techniques Explained</b></summary>
<br>

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **K-Fold** | Split into k folds; each used as validation once | Balanced, efficient | Random split |
| **Stratified K-Fold** | Maintains target distribution per fold | Reduces bias | Needs binning for regression |
| **Time Series Split** | Training always before validation | No data leakage | Less training data |
| **LOOCV** | Each sample = validation set | Minimal bias | O(n²) expensive |

</details>

<details>
<summary><b>Q4. Why are tree-based models less sensitive to feature scaling?</b></summary>
<br>

Decision trees split on **thresholds** — only the relative ordering of values matters. Whether area is in range `[500, 3776]` or `[0, 1]` doesn't change the optimal split. Unlike **Ridge/Lasso/SVR** which depend on Euclidean distances and are magnitude-sensitive.

</details>

<details>
<summary><b>Q5. Bias–Variance Tradeoff</b></summary>
<br>

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias   → Underfitting → Model too simple → e.g., Linear Regression on complex data
High Variance → Overfitting → Model memorises noise → e.g., Deep Decision Tree (no depth limit)
Sweet Spot  → Random Forest with controlled depth → Best generalization
```

</details>

---

## 🤖 Models Implemented

| Model | Part | Key Params | Scaling? | Test R² | Status |
|-------|------|-----------|----------|---------|--------|
| Ridge Regression (L2) | C | α = 10 | ✅ Yes | 0.9197 | ✅ Good |
| Lasso Regression (L1) | C | α = 10,000 | ✅ Yes | 0.9198 | ✅ Good |
| Decision Tree | E | depth=8, min_leaf=20 | ❌ No | 0.9071 | ✅ Controlled |
| **🏆 Random Forest** | **E** | **100 trees, depth=10** | **❌ No** | **0.9274** | **⭐ BEST** |
| SVR Linear | F | C = 1.0 | ✅ Yes | –5.23 | ❌ Poor fit |
| SVR RBF | F | C=100, γ=0.01, ε=0.1 | ✅ Yes | ≈0.00 | ⚠️ Small sample |

---

## 🏆 Performance Results

### Model Comparison Table

| Rank | Model | R² Score | RMSE (₹M) | MAE (₹M) | Overfit? |
|:----:|-------|:--------:|:---------:|:--------:|:--------:|
| 🥇 1 | **Random Forest** | **0.9274** | **2.32** | **1.68** | ✅ Good |
| 🥈 2 | Lasso (L1) | 0.9198 | 2.54 | 1.95 | ✅ Good |
| 🥉 3 | Ridge (L2) | 0.9197 | 2.54 | 1.94 | ✅ Good |
| 4 | Decision Tree | 0.9071 | 2.68 | 1.97 | ⚠️ depth>10 |
| 5 | SVR RBF | ≈0.00 | — | — | 🔴 Underfit |
| 6 | SVR Linear | –5.23 | — | — | 🔴 Poor |

### R² Score Visual

```
Random Forest  ████████████████████████████████████████████ 0.9274 ⭐
Lasso  (L1)    ███████████████████████████████████████████  0.9198
Ridge  (L2)    ███████████████████████████████████████████  0.9197
Decision Tree  ██████████████████████████████████████████   0.9071
SVR RBF        █                                            ≈0.00
SVR Linear     ░ (negative)                                 –5.23
```

### Cross-Validation Scores

| Strategy | Mean R² | Std Dev | Notes |
|----------|:-------:|:-------:|-------|
| K-Fold (k=5) | **0.9163** | ±0.004 | Most reliable |
| Stratified K-Fold | **0.9164** | ±0.004 | Similar to KFold (uniform target) |
| Time Series Split | **0.9157** | — | Slight drop (temporal pattern) |
| LOOCV | Per-sample | High variance | Unbiased but slow |

---

## 📈 Visualizations (10 Charts)

> All charts generated via Matplotlib & Seaborn. See `charts/` folder or the executed notebook.

| # | Chart | What it Shows |
|---|-------|---------------|
| 📊 01 | `01_eda_distributions.png` | Distribution of all 8 features + target |
| 🔥 02 | `02_correlation_heatmap.png` | Feature-to-feature + feature-to-target correlation |
| 🔵 03 | `03_feature_vs_target.png` | Scatter plots with trend lines for each feature vs price |
| 📉 04 | `04_regularization_tuning.png` | Ridge & Lasso Train/Val R² across alpha values |
| ⚖️ 05 | `05_ridge_lasso_coefficients.png` | Coefficient bar charts — sparsity comparison |
| 🔁 06 | `06_cross_validation.png` | All 4 CV strategies — bar chart + box plot |
| 🌲 07 | `07_tree_models.png` | DT depth overfitting analysis + RF feature importance |
| ⚡ 08 | `08_svr_results.png` | SVR Linear vs RBF — actual vs predicted scatter |
| 🏆 09 | `09_all_models_comparison.png` | R², RMSE, MAE for all 6 models side-by-side |
| 🔬 10 | `10_diagnostics_bias_variance.png` | Best model residuals + Train/Test R² + Learning Curve |

---

## 📁 Project Structure

```
🏠 robust-regression-engine/
│
├── 📓 Robust_Regression_Engine.ipynb      ← Pre-executed notebook (all outputs)
│
├── 📊 data/
│   ├── house_price.csv                    ← Raw dataset (3,800 × 12)
│   └── final_model_predictions.csv        ← ✅ Predictions from all 6 models
│
├── 📈 charts/
│   ├── 01_eda_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_vs_target.png
│   ├── 04_regularization_tuning.png
│   ├── 05_ridge_lasso_coefficients.png
│   ├── 06_cross_validation.png
│   ├── 07_tree_models.png
│   ├── 08_svr_results.png
│   ├── 09_all_models_comparison.png
│   └── 10_diagnostics_bias_variance.png
│
└── 📄 README.md                           ← This file
```

---

## 📋 Summary Report

### 🔒 Regularization
- **Ridge (L2)** best alpha = **10** → R² = 0.9197, all 11 features retained
- **Lasso (L1)** best alpha = **10,000** → R² = 0.9198, same features (dataset not high-dim enough to zero any)
- Both regularized models **significantly more stable** than unregularized baseline

### 🔁 Cross-Validation Insights
- **KFold & Stratified KFold** are almost identical → target distribution is uniform
- **Time Series Split** slightly lower → temporal dependencies reduce generalization slightly
- **LOOCV** gives per-sample unbiased estimates but is computationally expensive

### 🌲 Tree Models
- Decision Tree at **depth > 10** → overfitting (train R² ≈ 0.98, test drops to 0.88)
- Controlled at **depth = 8, min_leaf = 20** → test R² = 0.9071
- **Random Forest (100 trees)** → test R² = **0.9274**, RMSE = ₹2.32M → **Best Model**
- Ensemble reduced variance by **~20%** vs single decision tree

### ⚡ SVR Analysis
- **SVR Linear** → negative R² on full dataset → not suitable for large-scale regression without careful tuning
- **SVR RBF** → trained on 800-sample subset → underfits test set; needs full data + GridSearch for good results

### 🏢 Business Interpretation
- `area_sqft` and `location_score` are the **top price drivers**
- `crime_rate_index` and `distance_city_km` **negatively** impact price
- `near_metro` and `near_school` add **moderate premiums**
- **Random Forest** can be deployed as the production pricing model
- At ₹2.32M RMSE on ₹20.7M average price → **~11% relative error** — acceptable for real estate estimates

---

## 👤 Author & Contact

<div align="center">

| | |
|---|---|
| **Name** | Ghost (Patel Dhairya) |
| **Institute** | 🏫 Red and White Skill Education (RWSkill) |
| **GitHub** | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| **LinkedIn** | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| **Instagram** | [@ghost_6927](https://instagram.com/ghost_6927) |

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram"/>
</a>

</div>

---

## 📜 License

```
MIT License

Copyright (c) 2024 Ghost (Patel Dhairya) — Red and White Skill Education

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software.
```

---

<div align="center">

**[⬆ Back to Top](#-robust-regression-engine)**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with ❤️ by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2024*

</div>
