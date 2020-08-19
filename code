import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import random
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot



train = pd.read_csv('/content/train.txt', sep=",", header=0)
test = pd.read_csv('/content/test.txt', sep=",", header=0)
#converting to numerical features
number = LabelEncoder()
train.iloc[:,5] = number.fit_transform(train.iloc[:,5].astype('str'))
train.iloc[:,6] = number.fit_transform(train.iloc[:,6].astype('str'))
train.iloc[:,11] = number.fit_transform(train.iloc[:,11].astype('str'))

test.iloc[:,5] = number.fit_transform(test.iloc[:,5].astype('str'))
test.iloc[:,6] = number.fit_transform(test.iloc[:,6].astype('str'))
test.iloc[:,11] = number.fit_transform(test.iloc[:,11].astype('str'))

y = train.iloc[:,19]
X = train.iloc[:,:19]

#class imbalance: 165 to 9835
ones = [j for j in range(y.shape[0]) if y[j]==1]

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

#training
batch_size = 250
for epoch in range(10):
    order = np.random.permutation(X.shape[0])
    for start_index in range(0,X.shape[0], batch_size):
        index = order[start_index:start_index+batch_size]
        batch_indexes = index.tolist() + ones
        X_batch = train.iloc[batch_indexes,:19]
        y_batch = train.iloc[batch_indexes,19]
        model.fit(X_batch, y_batch)
        pred = model.predict_proba(X)
        arg = [pred[j,:].argmax() for j in range(pred.shape[0])]
        auc = roc_auc_score(y,arg)

print(auc)
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y,arg)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='ROC_AUC')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

pred = pd.Series(model.predict_proba(test)[:,1]).round(4)
