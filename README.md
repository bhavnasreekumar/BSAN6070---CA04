# CA-04 Ensemble Models Income Classification

## Assignment Overview

This project is part of **BSAN 6070 – Intro to Machine Learning** and focuses on implementing and comparing four ensemble learning models to predict an individual’s income category (`<=50K` or `>50K`).

The objective of this assignment is to evaluate how model performance changes as the number of estimators increases and to compare **Accuracy** and **AUC** across multiple ensemble methods.

The four ensemble models implemented:

- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  

---

## Technologies & Packages Used

This analysis was conducted using **Python** in a Jupyter Notebook environment. The following libraries were used:

- `pandas` – Data manipulation and preprocessing  
- `numpy` – Numerical computations  
- `matplotlib` – Performance visualization  
- `sklearn` – Machine learning modeling  
  - `RandomForestClassifier`  
  - `AdaBoostClassifier`  
  - `GradientBoostingClassifier`  
  - `accuracy_score`  
  - `roc_auc_score`  
- `xgboost` – `XGBClassifier`  

---

## Steps in the Analysis

### **1: Data Preparation**

The cleaned and discretized dataset from CA03 was reused.

- Continuous variables were already binned  
- Categorical variables were encoded  
- A training/test flag was used to split the dataset  
- One-hot encoding was applied to categorical variables  
- Features were converted to NumPy arrays for XGBoost compatibility  

This ensured consistency with CA03 while adapting the data for ensemble modeling.

---

### **2: Train-Test Split**

The dataset was split using the provided training indicator column to ensure consistent evaluation across all models.

---

### **3: Model Training**

Each ensemble model was trained using varying numbers of estimators:

```python
n_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
```

For each model:

- Train the model on the training set  
- Predict on the test set  
- Compute:
  - **Accuracy**  
  - **AUC (Area Under the ROC Curve)**  
- Store results for comparison  

---

### **4: Performance Visualization**

For each model, two performance curves were generated:

- **Accuracy vs `n_estimators`**  
- **AUC vs `n_estimators`**  

These plots were used to analyze:

- Initial performance improvement  
- Plateau behavior  
- Diminishing returns from additional estimators  
- Stability of model performance  

---

### **5: Optimal Estimator Selection**

For each model, the best number of estimators was identified separately for:

- **Accuracy**
- **AUC**

A comparison table was created summarizing:

- Best `n` (Accuracy)  
- Best Accuracy  
- Best `n` (AUC)  
- Best AUC  

This allowed structured comparison of performance across all ensemble models.

---

## Final Model Performance Summary

Based on evaluation:

- **Gradient Boosting** achieved the highest overall Accuracy.  
- **Gradient Boosting** also achieved the highest AUC.  
- **AdaBoost** showed steady improvement as estimators increased.  
- **XGBoost** converged quickly and achieved strong performance with fewer estimators.  
- **Random Forest** improved initially but showed diminishing returns at higher estimator values.  

---

## Key Insights

- Increasing the number of estimators generally improves performance initially.  
- After a certain point, additional estimators provide minimal performance gains.  
- AUC continued improving longer than Accuracy in some models.  
- Boosting methods demonstrated stronger ranking performance compared to bagging.  
- Ensemble methods improved stability and predictive power compared to the single Decision Tree model from CA03.  

---

## Comparison to CA03

Compared to the Decision Tree model in CA03:

- Ensemble methods reduced variance and improved generalization.  
- AUC improved substantially relative to the base tree model.  
- Boosting models demonstrated stronger overall predictive consistency.  

---

## Authors

This project was completed by:

- **Bhavna Sreekumar**  
- **Jessica Shono Thai**
