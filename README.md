# 📊 Sales & Profit Prediction

A machine learning project that analyzes sales data to uncover trends and predicts profit using multiple regression models.

## 1. Data Understanding

- **Dataset:** `sales_data_10k.csv`
- **Target Variable:** `Profit ($)`
- **Features:**
  - `Sales ($)` — Revenue generated
  - `Category` — Type of product
  - `Region` — Geographic region
  - Any other relevant features present in the dataset

## 2. Objectives

- Perform **Exploratory Data Analysis (EDA)** to uncover sales and profit trends.
- Build and evaluate **4 machine learning models** to predict `Profit ($)`.
- Compare model performances and select the best model based on evaluation metrics.

## 3. Machine Learning Models

| Model | Description |
|---|---|
| **Linear Regression** | Basic model assuming a linear relationship |
| **Decision Tree Regressor** | Simple tree-based model that tends to overfit |
| **Random Forest Regressor** | Ensemble model to reduce overfitting |
| **Gradient Boosting Regressor** | Sequential model that improves weak learners |

## 4. Model Evaluation Metrics

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values — lower is better.
- **R-squared (R²):** Indicates how well the model explains the variability of the target — closer to 1 is better.

## 5. Project Structure

```
.
├── sales-profit-prediction-code.py   # Main analysis & modeling script
├── sales_data_10k.csv                # Dataset
└── README.md
```

## 6. Getting Started

```bash
git clone https://github.com/RAHULVANANI/<repo-name>.git
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn
python sales-profit-prediction-code.py
```

## 7. Future Improvements

- Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
- Add feature engineering to introduce more relevant features.
- Perform time-series analysis if time-based data is available.

## Author

**Rahul Vanani**
- [LinkedIn](https://www.linkedin.com/in/rahul-vanani-72ba60311/) · [GitHub](https://github.com/RAHULVANANI)


