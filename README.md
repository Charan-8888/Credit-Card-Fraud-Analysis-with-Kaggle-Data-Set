# Credit Card Fraud Detection — Mathematical & Statistical Analysis

**Course:** 25MT1306E — Mathematics for Data Science and Analytics
**Institution:** Koneru Lakshmaiah Education Foundation (KL University), Dept. of Mathematics
**Team:** Arabati Charan (2510030005) · Indra Karan Reddy (2510030148) · Nanda Kishor Reddy (2510030296)
**Guide:** Dr. XYZ

---

## 1. What This Project Is

This project applies core mathematical and statistical techniques — **not machine learning classification** — to a real-world dataset of credit card transactions, in order to mathematically quantify what separates fraudulent transactions from legitimate ones.

It answers one central question:

> *Can we statistically describe, measure, and predict patterns of credit card fraud using probability theory, hypothesis testing, and regression — without building a black-box ML classifier?*

---

## 2. Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), originally collected by the Machine Learning Group of ULB (Université Libre de Bruxelles) in collaboration with Worldline.

| Property | Value |
|---|---|
| Raw rows | 284,807 transactions |
| Cleaned rows | 283,726 (1,081 exact duplicates removed) |
| Columns | 31 |
| Time span | ~48 hours (Sept 2013, European cardholders) |
| Fraud rate | 0.17% (473 fraud vs 283,253 legitimate) |

**Columns:**
- `Time` — seconds elapsed since the first transaction in the dataset
- `V1`–`V28` — 28 anonymised numeric features. The bank ran the real transaction details (merchant, location, etc.) through **PCA (Principal Component Analysis)** before releasing the data publicly, so these columns are privacy-safe scrambled numbers, not directly interpretable
- `Amount` — transaction value in EUR
- `Class` — the label: `0` = legitimate, `1` = fraud

---

## 3. Project Pipeline (Modules M1–M6)

| Module | Name | What It Does |
|---|---|---|
| **M1** | Data Cleaning | Removes duplicate rows, checks for missing values, produces summary statistics |
| **M2** | Exploratory Data Analysis (EDA) | Histograms, class-imbalance bar chart, box plots, correlation heatmap, hourly trend chart |
| **M3** | Probability Analysis | Empirical probability of fraud, conditional probability by amount threshold, Normal distribution fitting |
| **M4** | Statistical Inference | Pearson correlation, Welch's independent t-test, 95% confidence interval |
| **M5** | Regression Modeling | Multiple Linear Regression predicting transaction `Amount` from PCA features |
| **M6** | Visualization | Actual-vs-predicted plots, residual plots, coefficient/feature-importance charts |

---

## 4. Tools & Libraries

| Library | Purpose |
|---|---|
| **Python 3.x** | Core programming language |
| **Pandas** | Loading, cleaning, and manipulating tabular data |
| **NumPy** | Numerical computation, array operations |
| **Matplotlib** | All charts — histograms, bar charts, line charts, scatter plots, heatmaps |
| **SciPy** | Statistical tests — t-test, Normal distribution, `linregress` |
| **Scikit-learn** | Multiple Linear Regression model, train/test split, evaluation metrics (R², MAE, RMSE) |

> **Note on the chart window:** When you run the script outside Jupyter, Matplotlib's `plt.show()` opens its own pop-up window (titled "Figure 1") with built-in zoom/pan/save controls. This is **Matplotlib's default viewer** — no custom GUI was built; it ships free with the library (backed by Tkinter or Qt, whichever is installed on your machine).

---

## 5. Key Results

### Data Cleaning (M1)
- 1,081 duplicate transactions removed → final dataset: 283,726 rows
- Zero missing values found across all 31 columns

### EDA (M2)
- **Extreme class imbalance**: 99.83% legitimate vs 0.17% fraud (598 : 1 ratio)
- `Amount` is heavily right-skewed (skewness = 16.9) — median €22.50 vs mean €88.47
- **V17** has the strongest correlation with fraud (`r = −0.31`)
- **V2** has the strongest correlation with `Amount` (`r = −0.53`)

### Probability Analysis (M3)
- `P(Fraud)` = 0.17% (≈ 1 in 600 transactions)
- `P(Fraud | Amount > €1,000)` = 0.88% — **5× higher** than the baseline rate
- Normal fit: `N(μ = 89.04, σ = 251.10)`

### Statistical Inference (M4)
- Welch's t-test: fraud transactions average **€123.87** vs legitimate **€88.41** (`t = 2.96, p = 0.003`) → statistically significant difference
- 95% Confidence Interval for mean transaction amount: **[€87.55, €89.39]**

### Regression Modeling (M5)
- Multiple Linear Regression (features: V1, V2, V3, V4, V14, V17, Time → predicting `Amount`)
- **R² = 0.40** (40% of Amount variance explained)
- MAE = €81.61, RMSE = €187.07
- `V2` is the strongest single predictor (β = −81.13)

---

## 6. Why This Matters

A naive model that labels every transaction "legitimate" would be 99.83% "accurate" — yet catch **zero fraud**. This is why the analysis relies on probability and statistical inference (not raw accuracy) to draw conclusions, and why class-imbalance-aware techniques (e.g., SMOTE) are recommended for any future classification model built on this data.

---

## 7. Files in This Project

| File | Description |
|---|---|
| `project_creditcard.py` | Full Python script (M1–M6) |
| `CreditCardFraud_MDS_Project.pdf` | Formal written project report with citations |
| `CreditCardFraud_Presentation.pptx` | 10-slide project presentation |
| `creditcard.csv` | Raw dataset (not included — download from Kaggle) |

---

## 8. How to Run

```bash
pip install pandas numpy matplotlib scipy scikit-learn
python project_creditcard.py
```

Make sure `creditcard.csv` is in the same folder as the script, or update the file path in the `pd.read_csv(...)` line.

---

## 9. Future Work

- Replace simple linear regression with classification models (Logistic Regression, Random Forest, XGBoost) for actual fraud prediction
- Apply **SMOTE** (Synthetic Minority Oversampling) to address the 598:1 class imbalance
- Explore unsupervised anomaly detection (Isolation Forest) for real-time fraud flagging

---

## 10. References

- Kaggle — Credit Card Fraud Detection Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE CIDM.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
