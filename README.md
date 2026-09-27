# Customer Segmentation with K-Means Clustering

## Business question

Can customer demographic, spending, purchasing-channel, and digital-engagement data be used to identify meaningful customer segments that a marketing team can act on?

This project builds an end-to-end **unsupervised machine-learning workflow** to answer that question.

### What makes this a data-science project?

Rather than stopping at `KMeans(n_clusters=5)`, the analysis focuses on the complete decision process:

- data-quality auditing
- reproducible feature engineering
- exploratory analysis
- distance-aware feature scaling
- model selection using WCSS and Silhouette Score
- cluster profiling
- PCA visualization
- post-clustering campaign analysis
- business interpretation
- limitations and validation strategy

---

## Dataset

The notebook works with a customer marketing dataset containing **2,240 records and 29 original variables**.

The data includes:

- demographics
- income
- household information
- product spending
- web/store/catalog purchases
- web visits
- campaign responses

The CSV is intentionally not committed by default. Place the dataset at:

```text
data/customer_segmentation.csv
```

and update the path if necessary.

---

## Key analytical decisions

### 1. Data quality

The original data contains:

- 24 missing `Income` values
- an extreme `Year_Birth` observation (`1893`)
- an extreme `Income` value (`666666`)

Missing income is median-imputed.

Potentially problematic birth-year and income observations are excluded from the **modeling sample** while remaining available in the original dataframe for audit.

These thresholds are project-specific and should be validated against the original source system before production use.

### 2. Reproducibility

Customer tenure is calculated using a fixed reference date rather than `pd.Timestamp('today')`.

This ensures that running the notebook next month produces the same feature values.

### 3. Feature engineering

The project creates:

- `Age`
- `Total_Children`
- `Total_Spending`
- `AcceptedAny`
- `Customer_Since`

### 4. Clustering features

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

The original analysis selected **K = 5** using the elbow analysis. The revised notebook recomputes the evaluation and reports the final silhouette score so the decision can be independently assessed.

---

## Results

The original notebook's five-cluster solution produced the following approximate profiles:

| Cluster | Avg. Age | Avg. Income | Avg. Spending | Web Purchases | Store Purchases |
|---:|---:|---:|---:|---:|---:|
| 0 | 48.4 | 31,453 | 100 | 2.15 | 3.09 |
| 1 | 59.7 | 58,751 | 850 | 7.83 | 7.65 |
| 2 | 47.2 | 77,670 | 1,287 | 4.65 | 8.58 |
| 3 | 67.3 | 42,582 | 161 | 2.45 | 3.88 |
| 4 | 70.4 | 73,675 | 1,212 | 4.66 | 8.57 |

The revised notebook recomputes these profiles after applying the documented data-quality rules. Therefore, the table above should be treated as the **baseline result from the original analysis**, not as a guaranteed reproduction of the revised model.

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
└── data/
    └── customer_segmentation.csv   # not included by default
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

## Limitations and future work

A stronger production version could:

- compare K-Means with hierarchical clustering, DBSCAN, and Gaussian Mixture Models
- perform cluster-stability analysis
- evaluate sensitivity to feature selection and outlier treatment
- use log/robust transformations for skewed monetary variables
- develop an RFM-style representation
- evaluate segment-level revenue or profit
- test segment-specific marketing interventions using A/B testing
- deploy customer-segment assignment through Streamlit

---

## Author

**Aditya Kumar Sony**

BSc Mathematics | Data Science & Machine Learning

Interests: Data Science, Machine Learning, Statistical Modeling, Mathematical Modeling, Applied Mathematics

---

## Portfolio note

This project is intended to demonstrate the complete reasoning process behind an unsupervised machine-learning solution: not only training a model, but also validating the data, justifying modeling choices, interpreting the output, and connecting analytical findings to business questions.
