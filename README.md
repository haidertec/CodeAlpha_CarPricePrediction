# CodeAlpha - Car Price Prediction

## 📌 Overview
End-to-end Machine Learning pipeline to predict used car prices based on features like present price, year, kilometers driven, fuel type, transmission, seller type, and brand. Built as part of CodeAlpha Data Science Internship — Task 3.

## 🎯 Task
CodeAlpha Data Science Internship - Task 3

## 🛠 Techniques Used

- **Feature Engineering** — created 4 new features:
  - `car_age` (derived from manufacturing year)
  - `kms_per_year` (average yearly distance)
  - `car_name_len` (length of car name)
  - `brand` (extracted from car name)

- **Data Preprocessing** — median/mode imputation + StandardScaler for numeric + OneHotEncoder for categorical

- **7 Regression Models**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
  - Decision Tree
  - Random Forest
  - Gradient Boosting

- **Hyperparameter Tuning** — RandomizedSearchCV with 5-Fold Cross-Validation

- **Ensemble Learning** — VotingRegressor + StackingRegressor with Ridge meta-learner

- **Evaluation Metrics** — R², RMSE, MAE, MAPE (test + cross-validation)

## 📊 Key Results

- **Best single model**: Random Forest / Gradient Boosting
- **Ensemble models**: Voting & Stacking delivered superior generalization
- **Top predictive features**:
  - `present_price` (strongest predictor)
  - `car_age`
  - `kms_driven`
  - `kms_per_year`

## 📈 Visualizations Generated

1. Feature correlation heatmap
2. Feature vs Price scatter plots (2×3 grid)
3. Price distribution histogram with KDE
4. Boxplots of price by categorical features
5. Actual vs Predicted + Residual plots
6. Model comparison chart (R² vs RMSE)
7. Permutation feature importance (Top 15)
8. Learning curve (training vs validation R²)
9. Mutual information feature ranking

## 🧰 Tech Stack

Python • Scikit-learn • Pandas • NumPy • Matplotlib • Seaborn • SciPy • Jupyter • Joblib

## 📂 Repository Structure
