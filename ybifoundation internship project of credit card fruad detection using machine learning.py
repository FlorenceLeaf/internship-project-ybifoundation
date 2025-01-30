# ybifoundation internship project of credit card fruad detection using machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

data = pd.read_csv("D:/My Files/Games/Elden Ring/Game/flutter_windows_3.27.2-stable/creditcard.csv")
print(data.head())
print("Dataset shape:", data.shape)
print("Dataset description:")
print(data.describe())

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlierFraction = len(fraud) / float(len(valid))
print("Outlier Fraction:", outlierFraction)
print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Transactions: {len(valid)}")

print("Amount details of the fraudulent transaction:")
print(fraud.Amount.describe())
print("Details of valid transaction:")
print(valid.Amount.describe())

corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.title("Correlation Matrix")
plt.show()

X = data.drop(['Class'], axis=1)
Y = data["Class"]
xData = X.values
yData = Y.values

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(xTrain, yTrain)

yPred = rfc.predict(xTest)

print("The model used is Random Forest Classifier")

acc = accuracy_score(yTest, yPred)
prec = precision_score(yTest, yPred)
rec = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
MCC = matthews_corrcoef(yTest, yPred)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-Score: {f1}")
print(f"Matthews Correlation Coefficient: {MCC}")

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'], annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

