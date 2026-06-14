# 🏠 House Prices ML Pipeline

> An end-to-end machine learning pipeline for predicting residential house prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset.

---

## 🚀 Overview

This project builds an end-to-end machine learning pipeline for predicting residential house prices using the Kaggle House Prices dataset.

The workflow includes:

- Data preprocessing
- Missing value handling
- Feature encoding
- Feature selection
- Model comparison
- Hyperparameter tuning
- Model evaluation

The final tuned XGBoost model achieved an R² score of **0.9243** on the test set.

---

---

## 📋 Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Models Evaluated](#models-evaluated)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Key Learnings](#key-learnings)

---

## 📦 Dataset

**Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

The dataset contains detailed information about residential properties including:

- Lot size and dimensions
- House quality and condition
- Basement features
- Garage information
- Neighborhood characteristics
- Sale price

> ⚠️ Dataset files are **not included** in this repository. Download them from Kaggle and place `train.csv` in the project root directory.

---

## 🔧 Data Preprocessing

### Missing Value Handling

Several domain-specific imputation strategies were applied:

| Feature Group | Strategy |
|---|---|
| `LotFrontage` | Neighborhood-wise median imputation |
| Basement features | Filled with `No Basement` |
| Garage features | Filled with `No Garage` |
| Pool features | Filled with `No Pool` |
| Fireplace features | Filled with `No Fireplace` |

### Feature Encoding

Categorical and ordinal features were converted to numerical representations using:

- **One-Hot Encoding** with `drop_first=True` to reduce multicollinearity

---

## 🎯 Feature Selection

Feature importance scores were extracted using **XGBoost**. Features with **zero importance** were dropped, and the model was retrained on the reduced feature set to improve generalization.

---

## 🤖 Models Evaluated

| Model | R² Score |
|---|---:|
| Decision Tree Regressor | 0.7798 |
| Random Forest Regressor | 0.8899 |
| **Tuned XGBoost Regressor** | **0.9243** |

---

## ⚙️ Hyperparameter Tuning

Optimization was performed using **RandomizedSearchCV** with **5-Fold Cross Validation**.

### Best Parameters

```python
{
    'subsample': 0.7,
    'n_estimators': 500,
    'min_child_weight': 5,
    'max_depth': 4,
    'learning_rate': 0.1,
    'colsample_bytree': 0.7
}
```

### Scores

| Metric | Score |
|---|---|
| Best CV Score | `0.8761` |
| Final Test Score | `0.9243` |

---

## 🏆 Results

The tuned XGBoost model achieved an **R² Score of 0.9243**, outperforming both Decision Tree and Random Forest baselines.

---

## 📁 Project Structure

```text
house-prices-ml-pipeline/
│
├── kaggle_house_price_prediction.py
├── data_description.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

---

## 💡 Key Learnings

- Missing value handling with domain-specific strategies
- Feature engineering and categorical encoding
- Model comparison and evaluation
- Overfitting analysis
- Feature importance analysis with XGBoost
- Hyperparameter tuning using `RandomizedSearchCV`
- Building a complete end-to-end ML workflow