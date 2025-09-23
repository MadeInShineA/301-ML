[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GS7DvEfm)
 # Week 2 - Halfdays 2-3
This lab is devided into four main sections:
- Data exploration
- Data preparation
- Model training
- Evaluation

This represents a typical workflow in a machine learning project.
Then, in a fifth and last section, you will be asked to investigate the Support Vector Machine (SVM) algorithm and to add it to your workflow.

### Input
We provide you with a [dataset](https://www.kaggle.com/c/titanic/data) and a structured jupyter notebook that will guide you throw the different parts.

### Expected Outcomes
A Jupyter notebook to be submitted via GitHub classroom.
Please make sure to:
- Explain the goal of your code (i.e., what you are doing and *why*)
- Comment on the results
- Provide direct answers to **all** the questions posed in the notebook

We *expect* that you take the time to understand the concepts mentioned below, the code you write and the results you obtain.

**Golden rule**: You should be able to explain everything you do and why you do it.

*NOTE: the concepts noted below between brackets (e.g., [ROC curve]) are considered optional content.*

### Main Tools
NumPy, pandas, scikit-learn

## Mission 1 - Data Exploration (Week 2 - Halfday 2)
- **Problem**: Gaining a deep understanding of the available dataset: use case [Titanic dataset](https://www.kaggle.com/c/titanic/data)
- **ML Concepts**: Metadata, categorical/numerical features, correlation, features, parameters, hyperparameters, labels, classification Vs. regression, supervised Vs. unsupervised learning, balanced/unbalanced dataset, missing values, outliers.

## Mission 2 - Data Preparation (Week 2 - Halfday 3)
- **Problem**: Prepare the data for training a machine learning model
- **ML Concepts**: Imputation, normalization, standardization, one-hot encoding, feature engineering, feature selection.

## Mission 3 - Model training (Week 2 - Halfday 3-4)
- **Problem**: Train a machine learning model on the prepared data. You will start with a simple k-nn model. Then you will add a Support Vector Machine (SVM) and compare the results.
- **ML Concepts**: k-nn, cross-validation, hyperparameters fine-tuning (grid search, random search, [optuna]), learning curves, overfitting Vs. underfitting.

## Mission 4 - Evaluation (Week 2 - Halfday 4)
- **Problem**: Evaluate the trained model on a test set
- **ML Concepts**: Accuracy, precision, recall, F1-score, confusion matrix, [ROC curve], [AUC].

## Mission 5 - Support Vector Machine (Week 2 - Halfday 4-5)
- **Problem**: Investigate the SVM algorithm and add it to your workflow.
- **ML Concepts**: SVM, kernel trick, hyperplane, support vectors, margin.

### Additional task (not in the notebook):

Create 5-10 slides presenting SVM. The content of your slides should enable you to explain the following concepts:
- Hyperplane of separation.
- Soft margin.
- Support vector.
- Kernel (polynomial and RBF).
- Understand and explain the kernel trick, why it a "trick", and why it's useful.
- Discuss possible approaches for multiclass classification using SVM.
- What is one-class SVM and what is it used for?
- How does SVM select the best hyperplane of separation?
- SVM (without considering the kernel) has only one hyperparameter. What is its name, and what does it represent?

*NOTE: cite the sources that you used to prepare your presentation.*
