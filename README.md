# predictive-modeling-pipeline
7 classification models on 100K+ records with 10-fold cross-validation
# Predictive Modeling Pipeline

> End-to-end analytics pipeline on a 100K+ record dataset — 7 classification models evaluated with 10-fold cross-validation, achieving ~77% accuracy.

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Tools](https://img.shields.io/badge/Tools-Python%20%7C%20R%20%7C%20caret-276DC3)

---

## 📌 Project Overview

This project demonstrates a full analytics pipeline from business objective definition to model evaluation. Working with a dataset of over 100,000 records, the goal was to translate business KPIs into a measurable classification problem, engineer predictive features, and identify the best-performing model through rigorous cross-validation.

---

## 🎯 Objectives

- Translate business objectives into measurable **KPIs and analytical workflows**
- Design and engineer **20+ predictive features** from raw data
- Build and compare **7 classification models** using best practices
- Manage deliverables using **Agile/Jira** sprint structure
- Document findings clearly for both technical and non-technical audiences

---

## 🤖 Models Evaluated

| Model | Accuracy | Notes |
|---|---|---|
| Naive Bayes | ~77% | Best generalization performance |
| Logistic Regression | — | Baseline benchmark |
| Decision Tree | — | Interpretability focus |
| Random Forest | — | Ensemble approach |
| KNN | — | Distance-based |
| SVM | — | High-dimensional performance |
| LDA | — | Linear discriminant |

> All models evaluated using **10-fold cross-validation** to prevent overfitting.

---

## 🔧 Pipeline Steps

1. **Business Understanding** — Defined KPIs and success metrics aligned to business goals
2. **Data Ingestion** — Loaded and explored 100K+ record dataset
3. **Data Preprocessing** — Handled missing values, outliers, encoding
4. **Feature Engineering** — Created 20+ features from raw attributes
5. **Model Training** — Trained 7 classifiers with consistent preprocessing
6. **Evaluation** — 10-fold CV; compared accuracy, precision, recall, F1
7. **Documentation** — Sprint-tracked deliverables via Jira

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Languages | Python, R |
| ML Libraries | caret, MASS, glmnet |
| Data Processing | Pandas, NumPy, dplyr |
| Project Management | Jira |
| Environment | Jupyter Notebook |

---

## 📁 Repository Structure

```
predictive-modeling-pipeline/
├── data/
│   └── dataset_sample.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── models/
│   └── model_comparison_results.csv
├── reports/
│   └── final_report.pdf
└── README.md
```

---

## 📈 Key Results

- **Naive Bayes** achieved the strongest generalization with ~77% accuracy across folds
- Feature engineering contributed meaningfully to model lift over baseline
- Sprint-based workflow ensured structured delivery and clear documentation at each stage

---

## 👩‍💻 Author

**Vaishnavi Perka** — [LinkedIn](https://www.linkedin.com/in/vaishnavi-perka) · [Portfolio](https://vaishnaviperka.github.io)
