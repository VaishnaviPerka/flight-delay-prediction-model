# Flight Delay Prediction

> A comprehensive end-to-end machine learning pipeline to classify U.S. domestic flight delays using 12 classification models — achieving 77% accuracy with Naive Bayes as the best-performing model.

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Language](https://img.shields.io/badge/Language-R-276DC3)
![Dataset](https://img.shields.io/badge/Dataset-BTS%20On--Time%20Performance-orange)
![Models](https://img.shields.io/badge/Models-12%20Classifiers-blue)

---

## 📌 Project Overview

Flight delays pose significant operational and financial challenges across the aviation industry. This project builds a predictive classification system to determine whether a U.S. domestic flight will experience a departure delay of **15 minutes or more** (`DepDel15`), using a large-scale dataset from the **U.S. Bureau of Transportation Statistics (BTS) On-Time Performance** database.

> **Co-authored with:** Hariharan Jothimani

---

## 🎯 Objectives

- Build a full preprocessing pipeline for a high-dimensional, heterogeneous aviation dataset
- Train and compare **12 classification models** across linear, discriminant, penalized, kernel-based, and probabilistic families
- Use **10-fold cross-validation** with Cohen's Kappa as the primary metric (to handle class imbalance)
- Identify the best-performing model and analyze its key predictors

---

## 📊 Dataset

- **Source:** U.S. Bureau of Transportation Statistics — On-Time Performance Database
- **Scope:** Filtered to top 5 busiest origin airports for computational feasibility
- **Training observations:** 155,945
- **Target variable:** `DepDel15` — Binary (Delay / NoDelay)
- **Predictors:** 14–18 features after preprocessing

### Key Features Used

| Type | Variables |
|---|---|
| Categorical | Airline, Origin, Dest, Quarter, Month, DayOfWeek |
| Numerical | AirTime, CRSElapsedTime, Distance, CRSDepTime |
| Engineered | CRSDepTime_sin, CRSDepTime_cos (cyclic encoding), DepTimeBlk_num |
| Binary | Cancelled, Diverted |

---

## ⚙️ Preprocessing Pipeline

1. **Sample size selection** — Filtered to top 5 busiest airports to manage scale
2. **Variable removal** — Dropped 40+ redundant/leakage columns (IDs, timestamps, arrival fields)
3. **Cyclic time encoding** — Converted `CRSDepTime` to sine/cosine components to preserve time continuity
4. **Missing value handling** — Complete-case removal via `na.omit()`
5. **Dummy encoding** — Quarter, Month, DayOfWeek expanded to binary indicators
6. **Mixed effect encoding** — `step_lencode_mixed()` for high-cardinality categoricals (Airline, Origin, Dest)
7. **Near-zero variance removal** — Used `nearZeroVar()` to eliminate low-signal predictors
8. **Box-Cox transformation** — Applied to AirTime, CRSElapsedTime, Distance to reduce skewness
9. **Correlation removal** — Removed predictors with correlation > 0.90 for model families requiring it
10. **Train/test split** — Stratified 80/20 split using `createDataPartition()`

---

## 🤖 Models & Results

Each model family received a tailored version of the preprocessed dataset based on its assumptions:

| Preprocessing Group | Models |
|---|---|
| Box-Cox + Correlation Removal | Logistic Regression, LDA, QDA, MDA |
| Box-Cox Only | PLS-DA, Penalized (glmnet), RDA |
| Correlation Removal Only | Neural Network, Naive Bayes, FDA |
| No Transformation | SVM (Radial), KNN |

### Training Performance (10-fold CV)

| Model | Training Kappa | Training Accuracy | Best Parameters |
|---|---|---|---|
| **Naive Bayes** ⭐ | **0.1837** | **0.7924** | fL=0, usekernel=FALSE, adjust=0.5 |
| QDA | 0.1775 | 0.7849 | — |
| RDA | 0.1706 | 0.7991 | gamma=1, lambda=0.75 |
| FDA | 0.1560 | 0.8001 | degree=2, nprune=5 |
| KNN | 0.1503 | 0.7996 | k=9 |
| Neural Network | 0.1341 | 0.8013 | size=5, decay=0.01 |
| MDA | 0.1304 | 0.7991 | subclasses=4 |
| SVM Radial | 0.1225 | 0.8010 | sigma=0.018, C=0.5 |
| LDA | 0.0908 | 0.7991 | — |
| Logistic | 0.0670 | 0.7976 | — |
| Penalized | 0.0635 | 0.7973 | alpha=0.25, lambda=0.001 |
| PLS-DA | 0.0111 | 0.7935 | ncomp=3 |

### Test Set Evaluation (Top 2 Models)

| Model | Test Kappa | Test Accuracy |
|---|---|---|
| **Naive Bayes** ⭐ | **0.1745** | 0.7719 |
| QDA | 0.1679 | 0.7822 |

> **Naive Bayes** was selected as the final model based on its higher Test Kappa — a more reliable metric under class imbalance than accuracy alone.

---

## 🏆 Best Model — Naive Bayes

### Confusion Matrix Summary

| Metric | Value |
|---|---|
| Accuracy | 77.19% |
| Kappa | 0.1745 |
| Sensitivity (NoDelay) | 0.9129 |
| Specificity (Delay) | 0.2337 |
| Balanced Accuracy | 0.5733 |

The model excels at identifying on-time flights (91% sensitivity) but faces challenges detecting actual delays — a known difficulty in class-imbalanced aviation datasets.

### Top 10 Predictors (Variable Importance)

| Rank | Predictor | Importance |
|---|---|---|
| 1 | Airline | 10.9% |
| 2 | CRSDepTime_sin | 10.6% |
| 3 | DepTimeBlk_num | 10.6% |
| 4 | Dest | 10.4% |
| 5 | Origin | 10.2% |
| 6 | CRSDepTime_cos | 9.8% |
| 7 | CRSElapsedTime | 9.8% |
| 8 | DayOfWeek_X5 | 9.4% |
| 9 | DayOfWeek_X2 | 9.2% |
| 10 | DayOfWeek_X4 | 9.2% |

Airline carrier, departure time, and route (origin/destination) were the strongest delay predictors — consistent with known aviation delay patterns.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | R |
| ML Framework | caret |
| Libraries | MASS, mda, klaR, nnet, glmnet, kernlab, tidymodels, embed, recipes |
| Data Processing | dplyr, tidymodels, recipe steps |
| Validation | 10-fold cross-validation, Cohen's Kappa |

---

## 📁 Repository Structure

```
flight-delay-prediction/
├── Final_R_code.R          # Complete R code for preprocessing + all 12 models
├── FINAL_REPORT.pdf        # Full project report with analysis and results
├── README.md
└── data/
    └── dataset_note.md     # Instructions to download BTS dataset
```

---

## 🚀 How to Run

```r
# Install required packages
install.packages(c("caret", "dplyr", "tidymodels", "embed", "MASS",
                   "mda", "klaR", "nnet", "glmnet", "kernlab", "recipes"))

# Update the dataset path in Final_R_code.R
path <- "your/path/to/dataset.csv"

# Run the full script
source("Final_R_code.R")
```

> The dataset is sourced from the [BTS On-Time Performance Database](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD). Download and update the path before running.

---

## 🔮 Future Improvements

- [ ] Apply SMOTE or cost-sensitive learning to address class imbalance and improve delay detection (Specificity)
- [ ] Integrate real-time weather data as additional predictors
- [ ] Explore ensemble methods (stacking, boosting) for further performance gains
- [ ] Build a Shiny dashboard for interactive delay prediction

---

## 📜 References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
- Federal Aviation Administration (FAA). (2023). *Airline On-Time Performance Data*. U.S. DOT.

---

## 👩‍💻 Authors

**Vaishnavi Perka** — [LinkedIn](https://www.linkedin.com/in/vaishnavi-perka) · [Portfolio](https://vaishnaviperka.github.io)  
**Hariharan Jothimani**
