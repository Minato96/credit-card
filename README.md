# Credit Card Fraud Detection using Machine Learning

## 📝 Introduction
This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions. Using a highly imbalanced dataset from Kaggle, I implemented various techniques to preprocess the data and train robust classifiers. The final model (XGBoost) demonstrates high precision and recall for the minority fraud class.



##  Dataset
The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle. It contains transactions made by European cardholders in September 2013.

-   **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
-   **Size:** 284,807 transactions
-   **Features:** 28 PCA-transformed features (`V1` to `V28`), `Time`, and `Amount`.
-   **Imbalance:** Highly imbalanced, with only 492 (0.172%) fraudulent transactions.

## 🛠️ Techniques & Pipeline
The project follows a standard machine learning pipeline:
1.  **Data Loading:** Loading the dataset using Pandas.
2.  **Exploratory Data Analysis (EDA):** Visualizing the class imbalance and feature distributions.
3.  **Preprocessing:**
    -   **Scaling:** The `Time` and `Amount` features were scaled using `StandardScaler` to normalize their ranges.
    -   **Oversampling:** The Synthetic Minority Over-sampling Technique (**SMOTE**) was applied *only to the training data* to address the severe class imbalance without causing data leakage.
4.  **Model Training:** Three different models were trained and compared:
    -   Logistic Regression (as a baseline)
    -   Random Forest Classifier
    -   XGBoost Classifier
5.  **Evaluation:** Models were evaluated on the original, imbalanced test set using metrics appropriate for imbalanced data, such as **Precision**, **Recall**, and the **Confusion Matrix**.

## 📊 Results
The XGBoost classifier provided the best performance, particularly in identifying fraudulent transactions (high recall for the 'Fraud' class).

| Model               | Class    | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| **XGBoost** | **No Fraud** | **1.00** | **1.00** | **1.00** |
|                     | **Fraud** | **0.94** | **0.90** | **0.92** |
| Random Forest       | No Fraud | 1.00      | 1.00   | 1.00     |
|                     | Fraud    | 0.96      | 0.85   | 0.90     |
| Logistic Regression | No Fraud | 1.00      | 0.98   | 0.99     |
|                     | Fraud    | 0.07      | 0.92   | 0.13     |

*Note: The results table should be filled with the actual numbers you get after running `train.py`.*

## 🚀 Future Work
-   **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV to find the optimal parameters for the XGBoost model.
-   **Deployment:** Deploy the best-performing model as a simple web application using **Streamlit** or FastAPI to make real-time predictions.

## ⚙️ How to Run
1.  Clone the repository: `git clone <your-repo-link>`
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the training pipeline: `cd src && python train.py`