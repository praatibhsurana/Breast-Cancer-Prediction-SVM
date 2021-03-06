## Brief
The project was carried out on the breast cancer dataset compiled for research. It can be found at: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) and also on [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

### Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits. | 
Missing attribute values: none |
Class distribution: 357 benign, 212 malignant

### Correlation Heatmap of the various parameters after basic EDA
![Correlation Heatmap](https://github.com/praatibhsurana/Breast-Cancer-Prediction-SVM/blob/master/corr_heatmap.png?raw=true)

### Model
A SVM Classifier was used. Preprocessing and EDA was carried out and the 26 best parameters that affected the prediction were chosen. A little bit of tweaking on the C parameter and use of rbf kernel yielded better results as compared to a linear kernel.
The scores obtained were as follows:
1) Accuracy = 0.93
2) Precision = 0.95
3) Recall = 0.74
4) F1-Score = 0.83

The score can be improved on further analysis and experimentation with various kernels and tweaking of 'C' and 'gamma' parameters.
