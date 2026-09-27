# Customer Segmentation with K-Means Clustering

## Business question

Can customer demographic, spending, purchasing-channel, and digital-engagement data be used to identify meaningful customer segments that a marketing team can act on?

This project builds an end-to-end **unsupervised machine-learning workflow** to answer that question.


---

## Dataset

The notebook works with a customer marketing dataset containing **2,240 records and 29 original variables** taken from Kaggle.

The data includes:

- demographics
- income
- household information
- product spending
- web/store/catalog purchases
- web visits
- campaign responses


## Key analytical decisions

### 1. Data quality

The original data contains:

- 24 missing `Income` values
- an extreme `Year_Birth` observation (`1893`)
- an extreme `Income` value (`666666`)

Missing income is median-imputed.

Potentially problematic birth-year and income observations are excluded from the **modeling sample** while remaining available in the original dataframe for audit.

These thresholds are project-specific and should be validated against the original source system before production use.



### 2. Feature engineering

The project creates:

- `Age`
- `Total_Children`
- `Total_Spending`
- `AcceptedAny`
- `Customer_Since`

### 3. Clustering features

The final model uses:

```text
Age
Income
Total_Spending
NumWebPurchases
NumStorePurchases
NumWebVisitsMonth
```

These represent demographic, economic, purchasing-channel, and digital-engagement dimensions.

---

## Modeling

K-Means is trained after standardizing the features with `StandardScaler`.

Candidate cluster counts are evaluated using:

- **WCSS / Elbow Method**
- **Silhouette Score**

The original analysis selected **K = 5** using the elbow analysis which also recomputes the evaluation and reports the final silhouette score so the decision can be independently assessed.

---

## Results

We have five-cluster solution obtained from analysis:

|Cluster|   Age |   Income |   Total_Spending |   NumWebPurchases |   NumStorePurchases |   NumWebVisitsMonth |
|------:|------:|---------:|-----------------:|------------------:|--------------------:|--------------------:|
|   0   |58.18  |  73360.7 |          1204.4  |              4.73 |                8.65 |                2.82 |
|   1   | 36.23 |  30760.3 |            94.86 |              2.13 |                3.04 |                7.01 |
|   2   |35.1   |  76071.4 |          1295.24 |              4.65 |                8.57 |                2.7  |
|   3   |54.21  |  42595.2 |           156.25 |              2.44 |                3.86 |                5.53 |
|   4   |47.36  |  58538.2 |           839.56 |              7.76 |                7.59 |                6.46 |  




### Interpretation

The original clusters indicate several distinct customer profiles, including:

- lower-income / lower-spending customers
- active multi-channel customers
- high-income / high-spending customers
- older lower-engagement customers
- older affluent high-spending customers

Cluster names are descriptive and do not represent ground-truth customer classes.

---

## Business use cases

The segments can be used to formulate hypotheses for:

- customer retention
- loyalty programs
- personalized promotions
- digital cross-selling
- reactivation campaigns
- channel-specific marketing
- high-value customer management

A production team should validate these hypotheses using controlled experiments rather than assuming that clustering itself proves business impact.

---

## Project workflow

```text
Customer Data
     ↓
Data Quality Audit
     ↓
Missing-Value Handling + Outlier Review
     ↓
Feature Engineering
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
StandardScaler
     ↓
K-Means Candidate Models
     ↓
Elbow + Silhouette Evaluation
     ↓
Final K-Means Model
     ↓
Cluster Profiling
     ↓
PCA Visualization
     ↓
Campaign Response Analysis
     ↓
Business Hypotheses
```

---

## Repository structure

```text
customer-segmentation/
│
├── README.md
├── customer-segmentation-portfolio.ipynb
├── requirements.txt
└── customer_segmentation.csv  
```

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---


## Author

**Aditya Kumar Sony**

BSc Mathematics | Data Science & Machine Learning

Interests: Data Science, Machine Learning, Statistical Modeling, Mathematical Modeling, Applied Mathematics

---

## Portfolio note

This project is intended to demonstrate the complete reasoning process behind an unsupervised machine-learning solution: not only training a model, but also validating the data, justifying modeling choices, interpreting the output, and connecting analytical findings to business questions.
