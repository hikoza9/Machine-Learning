import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

path = os.path.join(os.path.dirname(__file__))
data = pd.read_csv(path + '/dataset/train.csv')

scaler = StandardScaler()

feature_columns = data.columns[:-1]
target_columns = data.columns[-1]

feature = scaler.fit_transform(data[feature_columns].values)
target = data[target_columns].values

pca_n = 20
pca = PCA(n_components=pca_n)
pca_transform = pca.fit_transform(feature)
pca_df = pd.DataFrame(data = pca_transform, columns = ["Principal Components " + str(i+1) for i in range(pca_n)])

x_train, x_test, y_train, y_test = train_test_split(pca_df, target, test_size=0.25, random_state=24)

model = LogisticRegression()
model.fit(x_train, y_train)

#F-1 Score using Train Set
# model_predict = model.predict(x_train)
# model_conf_matrix = confusion_matrix(y_train, model_predict)
# model_acc_score = accuracy_score(y_train, model_predict)
# print("confussion matrix")
# print(model_conf_matrix)
# print("-------------------------------------------")
# print("Accuracy of Logistic Regression:",model_acc_score*100,'\n')
# print("-------------------------------------------")
# print(classification_report(y_train,model_predict))

#F-1 Score using Test Set
model_predict = model.predict(x_test)
model_conf_matrix = confusion_matrix(y_test, model_predict)
model_acc_score = accuracy_score(y_test, model_predict)
print("confussion matrix")
print(model_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Logistic Regression:",model_acc_score*100,'\n')
print("-------------------------------------------")
print(classification_report(y_test,model_predict))