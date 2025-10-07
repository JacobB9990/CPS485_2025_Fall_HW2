# Answers

## Underfitting
**Underfitting** is when a model has high bias and cannot learn the structure of the data, leading to poor performance on both training and testing sets. Simply, the model is just **too simple** to understand the data.

We used LogisticRegression (C=0.01, max_iter=1000) on two features (Glucose, BMI) with StandardScaler and an 80/20 train/test split (random_state=42). The C controls the strength of regularization. We chose a low regularization as it focuses on keeping the model simple to prevent overfitting. We also only used two features as the model cannot really see the whole picture per say.

We can see underfitting with our results:
```bash
Training Accuracy: 0.761
Testing Accuracy:  0.753
```
While these numbers are not super low. They are about much lower than other accuracy scores we got when using a higher C and more features. We could also mess with the splits to make out model less accurate. With a 10/90 split:
```bash
Training Accuracy: 0.645
Testing Accuracy:  0.656
```

It is easier to fix our case, as we changed the C and hid data away. In the real world it is not that simple. You can mess with the split, change models, or add more features.

## Overfitting
**Overfitting** is when a model is too complex and starts to memorize the training data instead of learning the real patterns. This usually gives very low training error but worse performance on the test set. Simply, the model is just **too detailed** for the data it has.

We used RidgeClassifier with polynomial features of increasing degree, StandardScaler, and an 80/20 train/test split (random_state=42, stratified). At first the polynomial degree was low, so the model was simple. As we increased the degree, the model got more parameters and became more flexible. Eventually, the number of parameters went past the number of training samples.

We can see overfitting and the double descent in the results: the training error kept dropping as the model got more complex, the test error first went up when the model was overfitting, and then it started going down again once the model had very high capacity. The graph shows this with training error in blue, test error in red, and the point where the number of parameters is about the same as the number of training samples marked with a black dashed line.

This shows that sometimes making a model very large can actually make it generalize better than a moderately overfitted model.

## Requirements & Run

**Requirements**:
- pandas
- numpy
- scikit-learn
- matplotlib

**Commands**:

```bash
% python3 overfitting.py
```

```bash
% python3 underfitting.py
```