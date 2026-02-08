# Diabetes Progression Prediction using Machine Learning

## Project Overview
This project focuses on predicting diabetes disease progression using supervised regression techniques applied to the standardized diabetes dataset from scikit-learn. 
The objective was to build an end-to-end machine learning pipeline, compare multiple regression models, select the optimal performer using cross-validated evaluation,
interpret the model, and persist the final trained pipeline for future inference.

---

## Dataset
- **Source:** scikit-learn Diabetes Dataset  
- **Samples:** 442  
- **Features:** 10 standardized clinical indicators  
- **Target:** Quantitative measure of diabetes progression one year after baseline  

The dataset contained **no missing values**, allowing direct progression to preprocessing and modeling.

---

## Methodology

### 1. Data Preparation
- Separated **features (X)** and **target (y)**
- Performed **train–test split** for unbiased evaluation

### 2. Preprocessing
- Applied multiple **feature scaling strategies**:
  - StandardScaler  
  - MinMaxScaler  

### 3. Model Training & Comparison
Evaluated multiple regression algorithms:

- Linear Regression  
- Lasso Regression  
- Random Forest Regressor  

Used **GridSearchCV** with **k-fold cross-validation** for:
- Hyperparameter optimization  
- Fair model comparison  

### 4. Evaluation Metrics
Since this is a **regression problem**, performance was measured using:

- **R² Score** → Variance explained by the model  
- **RMSE (Root Mean Squared Error)** → Prediction error in original units  

---

## Best Model Selection
The optimal configuration was:

**Lasso Regression + StandardScaler**

Reason:
- Achieved **lowest RMSE**
- Achieved **highest cross-validated R²**
- Performed **automatic feature selection** via coefficient shrinkage

---

## Model Interpretation
Coefficient analysis showed:

- **Strong positive influence:** BMI, s5, Blood Pressure  
- **Negative influence:** s3, sex, s1  
- **Zeroed features:** age, s2, s4  

This confirms **Lasso regularization** effectively removes less informative predictors.

---

## Final Training & Persistence
- Retrained the optimal pipeline on the **full dataset**
- Ensured **maximum learning from available data**
- Saved the trained preprocessing-model pipeline using **Joblib** for:
  - Reproducibility  
  - Efficient storage  
  - Future inference integration  

---

## Results
Typical performance range on this dataset:

- **R² ≈ 0.45 – 0.47**
- **RMSE ≈ 53**

This aligns with the **known performance ceiling** of the dataset due to:
- Limited sample size  
- Restricted clinical feature set  
- Intrinsic medical variability  

---

## Key Learnings
- Importance of **proper evaluation metrics for regression**
- Role of **feature scaling in linear models**
- Effectiveness of **regularization for feature selection**
- Difference between **model tuning (GridSearchCV)** and **final evaluation (cross-validation)**
- Necessity of **saving full ML pipelines for deployment**

---

## Future Work
Potential improvements include:

- Incorporating **additional real clinical features**
- Exploring **advanced boosting algorithms** (XGBoost, LightGBM)
- Training on **larger real-world medical datasets**
- Integrating the saved model into a **production-ready application**

---

## Tech Stack
- **Python**
- **NumPy, Pandas**
- **scikit-learn**
- **Matplotlib**
- **Joblib**

---

## Author
**Shiva Prasad**  
Aspiring Machine Learning & Data Science Engineer  
Focused on building practical, end-to-end ML solutions and real-world predictive systems.

---
