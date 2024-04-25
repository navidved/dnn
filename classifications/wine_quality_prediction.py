"""Exploratory Data Analysis (EDA) refers to the method of studying and exploring record sets to apprehend their predominant traits,
discover patterns, locate outliers, and identify relationships between variables.
EDA is normally carried out as a preliminary step before undertaking extra formal statistical analyses or modeling."""

# dataset: https://www.kaggle.com/datasets/rajyellow46/wine-quality?resource=download

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('classifications/ds/winequality.csv')

# look at the first five rows of the dataset.
print(df.head())

# explore the type of data present in each of the columns present in the dataset.
df.info()

# explore the descriptive statistical measures of the dataset.
print(df.describe().T)

# check the number of null values in the dataset columns wise.
print(df.isnull().sum())

# impute the missing values by means as the data present in the different columns are continuous values.
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
    
print(df.isnull().sum().sum())  

# draw the histogram to visualise the distribution of the data with continuous values in the columns of the dataset.
df.hist(bins=20, figsize=(10, 10))
plt.show()

# draw the count plot to visualise the number data for each quality of wine.
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

# There are times the data provided to us contains redundant features they do not help with increasing the model’s
# performance that is why we remove them before using them to train our model.
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr(numeric_only=True) > 0.7, annot=True, cbar=False)
plt.show()

# From the above heat map we can conclude 
# that the ‘total sulphur dioxide’ and ‘free sulphur dioxide‘ are 
# highly correlated features so, we will remove them.
df = df.drop('total sulfur dioxide', axis=1)

# Model Development
# splitting it into training and validation data 
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# We have a column with object data type as well let’s replace it with the 0 and 1 as there are only two categories.
df.replace({'white': 1, 'red': 0}, inplace=True)

# After segregating features and the target variable from the dataset we will split it into 80:20 ratio for model selection.
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
print(xtrain.shape, xtest.shape)

# Normalising the data before training help us to achieve stable and fast training of the model.
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# let’s train some state of the art machine learning model on it.
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))

# Model Evaluation
# From the above accuracies we can say that Logistic Regression 
# and SVC() classifier performing better on the validation data with 
# less difference between the validation and training data.
# Let’s plot the confusion matrix as well for the validation data using the Logistic Regression model. 
metrics.ConfusionMatrixDisplay.from_estimator(models[1], xtest, ytest)
plt.show()

# Let’s also print the classification report for the best performing model.
print(metrics.classification_report(ytest, models[1].predict(xtest)))

