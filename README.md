# Classification-Task-Breast-Cancer-Wisconsin-Dataset
Independent Machine Learning Project (Jupyter Notebook: Python) 

This project explores various supervised machine learning algorithms and evaluate their performance scores. This proejct focuses on classification, analyzing pros and cons and determing when to use each of them.


This project works with the Breast Cancer Wisconsin dataset. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. 

For more information, see https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

## Table of Contents

**Part 0: Understanding and Reshaping Datasets**
- Explore distributions of data, types of features, possible imbalances in classes
- Display the correlation between each data using function `seaborn` `violinplot` and `pairplot`
- Modify Datasets using `stratify` 

**Part 1: K-Nearest Neighbors**
- Fit the data using `sklearn` `KNeighborsClassifier`
- Explore various k values and identify the best-resulting k constant
- Display dependence on amounts of data using seaborn, altering amounts of data and drawing different split of training/testing data
- To prevent overfitting, use `SequentialFeatureSelector` to see which feature variables are important.

**Part 2: Linear Regression Model**
- Because linear regression is heavily based on weights and distributions of datasets, modify the data using `normalize`
- Explore various linear regression models -- `LinearRegression`, `LogisticRegression`, `RidgeCV`, `Perceptron`, and `SVM`
- Determine whether the given datasets are seperable by a linear model

**Part 3: Nonlinear Regression Model**
- `SVM`can be utilized for a non-linear regression by changing into different kernals 
- Develop tree-based models using `DecisionTreeClassifier` and `RandomForestClassifier`
- Train an artifical neural network using `MLPClassifier`

**Part 4: Neural Network**
- Define and train a simple Neural-Network with `PyTorch` and use it via `skorch` with `SciKit-Learn`. Bulid ANN.
