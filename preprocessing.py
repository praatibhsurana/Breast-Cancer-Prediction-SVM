import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import sklearn


def preprocess(data):
    # Annotating labels
    labels = []
    for i in range(len(data["diagnosis"])):
        if data["diagnosis"][i] == "M":
            labels.append(1)
        else:
            labels.append(0)

    # dropping non impacting columns as observed from heatmap
    data = data.drop(
        [
            "symmetry_se",
            "fractal_dimension_se",
            "smoothness_se",
            "fractal_dimension_mean",
            "diagnosis",
            "id",
            "Unnamed: 32",
        ],
        axis=1,
    )

    return data, labels


def train_test_split(data, labels):
    data = data.to_numpy()

    # Train and test data
    train = data[:450]
    train_labels = np.array(labels[:450])
    test = data[450:]
    test_labels = np.array(labels[450:])

    # Normalizing data
    train = sklearn.preprocessing.normalize(
        train, axis=1, norm="l2", copy=False, return_norm=False
    )
    test = sklearn.preprocessing.normalize(
        test, axis=1, norm="l2", copy=False, return_norm=False
    )

    return train, train_labels, test, test_labels


def final_preprocess(arr):
    arr = np.array(arr)

    arr = sklearn.preprocessing.normalize(
        arr, axis=1, norm="l2", copy=False, return_norm=False
    )

    return arr