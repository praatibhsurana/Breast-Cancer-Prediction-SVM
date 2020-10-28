import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import preprocessing as pre
import joblib

# Accessing dataset
data = pd.read_csv("data.csv")
# print(data.head())
print("Initial shape:", data.shape)
data = pd.DataFrame(data)

data, labels = pre.preprocess(data)
print(
    "Shape after preprocessing:", data.shape, "and length of target array:", len(labels)
)

train, train_labels, test, test_labels = pre.train_test_split(data, labels)
print("Train data:", train.shape, len(train_labels))
print("Test data:", test.shape, len(test_labels))

# SVM Classifier
clf = svm.SVC(kernel="rbf", C=1.0)  # rbf kernel

# Train the model using the training sets
clf.fit(train, train_labels)

# Predict the response for test dataset
y_pred = clf.predict(test)

print("Accuracy:", metrics.accuracy_score(test_labels, y_pred))

prec = metrics.precision_score(test_labels, y_pred)
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", prec)

rec = metrics.recall_score(test_labels, y_pred)
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", rec)

print("F-1 Score:", (2 * prec * rec) / (prec + rec))

# Saving model
filename = "pred.sav"
joblib.dump(clf, filename)