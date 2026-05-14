# Diabetes Prediction Analysis

This repository contains a comprehensive machine learning pipeline to predict diabetes using clinical datasets (e.g., PIMA Indians Diabetes Database and LMCH dataset). It encompasses data visualization, preprocessing, and the evaluation of multiple classification models, culminating in an advanced model stacking approach.

## Repository Structure

``` text
diabetes-prediction/
├── data/               # Dataset files
│   ├── raw/            # Original datasets
│   └── processed/      # Cleaned data
├── notebooks/          # Jupyter Notebooks for analysis
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Models_SVM_RF_DNN.ipynb
│   └── 03_Models_KNN_XGBoost_Stacking.ipynb
├── src/                # Reusable Python modules
│   ├── data_processing.py
│   └── evaluation.py
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Introduction

This project aims to reproduce the methodology presented in the paper [*Diabetes mellitus prediction and diagnosis from a data preprocessing and machine learning perspective*](https://doi.org/10.1016/j.cmpb.2022.106773) (Olisah et al., *Computer Methods and Programs in Biomedicine*, 2022). Building upon their foundation, this repository extends the research through an exploratory analysis of additional machine learning algorithms and advanced ensemble techniques, specifically model stacking.

## Methodology

### 1. Data Exploration & Preprocessing

-   **Exploratory Data Analysis (EDA)**: Visualized feature distributions (Glucose, Blood Pressure, Insulin, BMI, Age, etc.) using KDE and density plots to understand data skews.
-   **Missing Value Handling**: Zero values in continuous clinical features were treated as missing and imputed using the mean/median.
-   **Scaling & Encoding**: Applied `StandardScaler` for distance-based and gradient-based models. Target variables were properly encoded for classification tasks.

### 2. Modeling & Evaluation

We trained and evaluated several algorithms:

\- **K-Nearest Neighbors (KNN)**: Baseline distance-based model.

\- **Support Vector Machine (SVM)** & **Random Forest**: Analyzed tree-based vs. margin-based performance with GridSearchCV for hyperparameter tuning.

\- **Deep Neural Network (2GDNN)**: Constructed using TensorFlow/Keras to capture complex interactions with Dropout layers to prevent overfitting.

\- **XGBoost**: Gradient boosting framework, demonstrating robust performance on tabular data.

\- **Model Stacking (Ensemble)**: Combined XGBoost and KNN using a `LogisticRegression` meta-learner to maximize predictive accuracy, achieving superior generalizability.

## How to Run

1.  Clone the repository and install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

2.  Place the datasets in the `data/raw/` directory.

3.  Execute the notebooks in the `notebooks/` directory sequentially to reproduce the analysis.
