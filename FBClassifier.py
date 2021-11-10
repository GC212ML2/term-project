import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def brute_force(
    X:DataFrame,
    y:DataFrame,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
        LogisticRegression(solver="lbfgs", max_iter=100, multi_class="ovr", class_weight='balanced'),
        LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="ovr", class_weight='balanced'),
        GaussianNB(), GradientBoostingClassifier()
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    isCVShuffle = True,
):
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    - `X`: pandas.DataFrame
      - training dataset.
    - `y`: pandas.DataFrame
      - target value.
    - `scalers`: array
      - Scaler functions to scale data. This can be modified by user.
      - StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler as default.
    - `models`: array
      - Model functions to fitting data and prediction. This can be modified by user.
      - DecisionTreeClassifier, LogisticRegression, SVC as default with hyperparameters.
    - `k`: array
      - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].

    Returns
    ----------
    - `best_params_`: dictionary
      - `best_scaler_`: Scaler what has best score.
      - `best_model_`: Model what has best score.
      - `best_cv_k_`: k value in K-fold CV what has best score.
    - `best_score_`: double
      - Represent the score of the `best_params`.

    See Also
    ----------
    to_csv : Write DataFrame to a comma-separated values (csv) file.

    Examples
    ----------
    >>> pd.read_csv('data.csv')  # doctest: +SKIP
    """

    # Initialize variables
    maxScore = -1.0
    best_scaler = None
    best_model = None
    best_cv_k_ = None

    # Find best scaler
    for n in range(0, len(scalers)):
        X = scalers[n].fit_transform(X)

        # Find best model
        for m in range(0, len(models)):
            # Find best k value of CV
            for i in range(0, len(cv_k)):
                kfold = KFold(n_splits=cv_k[i], shuffle=isCVShuffle)
                score_result = cross_val_score(models[m], X, y, scoring="accuracy", cv=kfold)
                # if mean value of scores are bigger than max variable,
                # update new options(model, scaler, k) to best options
                if maxScore < score_result.mean():
                    maxScore = score_result.mean()
                    best_scaler = scalers[n]
                    best_model = models[m]
                    best_cv_k_ = cv_k[i]



    class res:
        best_params = {}

    res.best_params = {
        'best_scaler': best_scaler,
        'best_model' : best_model,
        'best_cv_k': best_cv_k_,
    }
    res.best_scaler = best_scaler
    res.best_model = best_model
    res.best_k = best_cv_k_

    res.best_score = maxScore

    # Return value with dictionary type
    return res


def plot_roc_curve(X, y, model, title):
    # for multiclass target
    # Calculate False Positive Rate, True Positive Rate
    X = model.best_scaler.fit_transform(X)
    y_unique, counts = np.unique(y, return_counts=True)
    y = label_binarize(y, classes=y_unique)
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = OneVsRestClassifier(model.best_model)
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    weighted_roc_auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")
    # Plot result
    plt.figure(figsize=(12,10))
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='cyan', label='Class 2 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='yellow', label='Class 3 vs Rest')

    plt.plot(fpr["macro"], tpr["macro"], color='r', label='ROC curve (area = %0.2f)' % weighted_roc_auc)
    plt.plot([0, 1], [0, 1], color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + str(title), fontsize=20)
    plt.legend()
    plt.show()

def clf_report(X, y, model):
    X = model.best_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = model.best_model
    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    report = classification_report(y_test, pred_test, zero_division=0, output_dict=True)
    weighted_avg = pd.DataFrame(report).T.loc['weighted avg', :]

    return weighted_avg

def auto_ml():
    print("auto ml")

