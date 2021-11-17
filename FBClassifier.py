
from math import nan
from matplotlib.pyplot import sca

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import random
import copy

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
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    is_cv_shuffle = True,
):
    """
    Brute Force Search
    ----------
    - Find the best parameter what has the best score.
    - This function use `Brute Force` method with memoization

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
      - DecisionTreeClassifier(gini, entropy) as default with hyperparameters.
    - `cv_k`: array
      - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].
    - `is_cv_shuffle`
      - To set shuffle or not in cross validation

    Returns
    ----------
    - `best_params`: dictionary type of results.
    - `best_scaler`: Scaler what has best score.
    - `best_model`: Model what has best score.
    - `best_cv_k`: k value in K-fold CV what has best score.
    - `best_score`: double
      - Represent the score of the `best_params`.
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

                kfold = KFold(n_splits=cv_k[i], shuffle=is_cv_shuffle)
                score_result = cross_val_score(models[m], X, y, scoring="accuracy", cv=kfold)

                # if mean value of scores are bigger than max variable,
                # update new options(model, scaler, k) to best options
                if maxScore < score_result.mean():
                    maxScore = score_result.mean()
                    
                    best_scaler = copy.deepcopy(scalers[n])
                    best_model = copy.deepcopy(models[m])
                    best_cv_k_ = copy.deepcopy(cv_k[i])



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

  
  
  
  
def random_search(
    X:DataFrame,
    y:DataFrame,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    is_cv_shuffle = True,
    thresh_score = None,
    max_iter = 50,
):
    """
    Random Search
    ----------
    - Find the best parameter what has the best score.
    - This function use `Random Search` method with memoization

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
      - DecisionTreeClassifier(gini, entropy) as default with hyperparameters.
    - `cv_k`: array
      - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].
    - `is_cv_shuffle`
      - To set shuffle or not in cross validation
    - `thresh_score`
      - Default is None. If, algorithm find the score what is higher than thresh_score, then stop and terminate searching.
    - `max_iter`
      - Default is 100. This is meaning that how many iterations in searching loop.

    Returns
    ----------
    - `best_params_`: dictionary type of results.
    - `best_scaler`: Scaler what has best score.
    - `best_model`: Model what has best score.
    - `best_cv_k`: k value in K-fold CV what has best score.
    - `best_score`: double
      - Represent the score of the `best_params`.
    """

    # 0. Calculate length of each paramenter
    scalers_len = len(scalers)
    models_len = len(models)
    cv_k_len = len(cv_k)

    # 0. Initialize max score point
    max_scalers_idx = 0
    max_models_idx = 0
    max_cv_k_idx = 0
    max_score = 0


    # 0. Create memorize table for memoization
    mem_table = [[[0 for col in range( cv_k_len ) ] for row in range( models_len )] for col in range( scalers_len ) ]

    for trial in range(0, max_iter):
        # 0. Pick arbitrary point (theta1 = p1)
        scalers_idx = random.randrange(0, scalers_len)
        models_idx = random.randrange(0, models_len)
        cv_k_idx = random.randrange(0, cv_k_len)

        # 2. Calculate score(J(theta)) of each theta(point)
        score = 0

        # Check mem_table if score already has been calculated
        if mem_table[scalers_idx][models_idx][cv_k_idx] != 0:
            score = mem_table[scalers_idx][models_idx][cv_k_idx]
        else:
            # if not, calculate score of theta
            if scalers[scalers_idx] != None:
                p1_X = scalers[scalers_idx].fit_transform(X)
            else:
                p1_X = X
            kfold = KFold(n_splits=cv_k[cv_k_idx], shuffle=is_cv_shuffle)
            score = cross_val_score(models[models_idx], p1_X, y, cv=kfold).mean()
            # 2-1. Memoization
            mem_table[scalers_idx][models_idx][cv_k_idx] = score
    

        # Save point parameter what have best score
        if max_score < score:
            max_scalers_idx = scalers_idx
            max_models_idx = models_idx
            max_cv_k_idx = cv_k_idx
            max_score = score

        # If, score get higher score than thresh, terminate gradient searching
        if thresh_score != None and max_score > thresh_score: break

        # print("Trial: ", end="")
        # print(trial)
        # print(p1.scalers_idx)
        # print(p1.models_idx)
        # print(p1.cv_k_idx)
        # print()

    class res:
        best_params = {}

    res.best_params = {
        'best_scaler': scalers[max_scalers_idx],
        'best_model' : models[max_models_idx],
        'best_cv_k': cv_k[max_cv_k_idx],
    }
    res.best_scaler = scalers[max_scalers_idx]
    res.best_model = models[max_models_idx]
    res.best_k = cv_k[max_cv_k_idx]

    res.best_score = max_score

    # Return value with dictionary type
    return res

  
  
  
  
def auto_ml(
    X:DataFrame,
    y:DataFrame,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    is_cv_shuffle = True,
    thresh_score = None,
    max_iter = 50,
):
    """
    Auto ML for Classifier
    ----------
    - Find the best parameter what has the best score.
    - This function use `Auto ML` method. This is similar to the Gradient Descent.
    - This function use memoization technique for faster calculation.

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
      - DecisionTreeClassifier(gini, entropy) as default with hyperparameters.
    - `cv_k`: array
      - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].
    - `is_cv_shuffle`
      - To set shuffle or not in cross validation
    - `thresh_score`
      - Default is None. If, algorithm find the score what is higher than thresh_score, then stop and terminate searching.
    - `max_iter`
      - Default is 50. This is meaning that how many iterations in searching loop.

    Returns
    ----------
    - `best_params`: dictionary type of results.
    - `best_scaler`: Scaler what has best score.
    - `best_model`: Model what has best score.
    - `best_cv_k`: k value in K-fold CV what has best score.
    - `best_score`: double
      - Represent the score of the `best_params`.
    """

    logo()
    
    # 0. Calculate length of each paramenter
    scalers_len = len(scalers)
    models_len = len(models)
    cv_k_len = len(cv_k)

    # 0. Create memorize table for memoization
    mem_table = [[[0 for col in range( cv_k_len ) ] for row in range( models_len )] for col in range( scalers_len ) ]
    
    # 0. Create point(theta) vector class
    class Point():
        scalers_idx = 0
        models_idx = 0
        cv_k_idx = 0
    
    # 0. Initialize gradient value
    gradient_theta1 = 0
    gradient_theta2 = 0
    gradient_theta3 = 0
    
    # 0. Pick arbitrary point (theta1 = p1)
    p1 = Point()
    p1.scalers_idx = random.randrange(0, scalers_len)
    p1.models_idx = random.randrange(0, models_len)
    p1.cv_k_idx = random.randrange(0, cv_k_len)

    # 0. Initialize max score point
    max_scalers_idx = 0
    max_models_idx = 0
    max_cv_k_idx = 0
    max_score = 0


    for trial in range(0, max_iter):

        # 1. Check previous gradient value of each theta
        #    and pick another arbitrary point (theta = p2)

        def check_gradient(target_gradient_theta, point_val, max_len):
            result = 0
            if target_gradient_theta > 0 and point_val+1 != max_len:
                # if point_val+1 == max_len => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(point_val + 1, max_len)

            elif target_gradient_theta < 0 and point_val != 0:
                # if point_val == 0 => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(0, point_val)
            
            else:
                result = random.randrange(0, max_len)

            return result


        p2 = Point()
        p2.scalers_idx = check_gradient(gradient_theta1, p1.scalers_idx, scalers_len)
        p2.models_idx = check_gradient(gradient_theta2, p1.models_idx, models_len)
        p2.cv_k_idx = check_gradient(gradient_theta3, p1.cv_k_idx, cv_k_len)



        # 2. Calculate score(J(theta)) of each theta(point)
        p1_score = 0
        p2_score = 0

        # Check mem_table if score already has been calculated
        if mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx] != 0:
            p1_score = mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx]
        else:
            # if not, calculate score of theta
            if scalers[p1.scalers_idx] != None:
                p1_X = scalers[p1.scalers_idx].fit_transform(X)
            else:
                p1_X = X
            kfold = KFold(n_splits=cv_k[p1.cv_k_idx], shuffle=is_cv_shuffle)
            p1_score = cross_val_score(models[p1.models_idx], p1_X, y, cv=kfold).mean()
            # 2-1. Memoization
            mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx] = p1_score

        if mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx] != 0:
            p2_score = mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx]
        else:
            if scalers[p1.scalers_idx] != None:
                p2_X = scalers[p2.scalers_idx].fit_transform(X)
            else:
                p2_X = X
            kfold = KFold(n_splits=cv_k[p2.cv_k_idx], shuffle=is_cv_shuffle)
            p2_score = cross_val_score(models[p2.models_idx], p2_X, y, cv=kfold).mean()
            # 2-1. Memoization
            mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx] = p2_score    


        # Save point parameter what have best score
        if p1_score > p2_score:
            if max_score < p1_score:
                max_scalers_idx = p1.scalers_idx
                max_models_idx = p1.models_idx
                max_cv_k_idx = p1.cv_k_idx
                max_score = p1_score

        if p1_score < p2_score:
            if max_score < p2_score:
                max_scalers_idx = p2.scalers_idx
                max_models_idx = p2.models_idx
                max_cv_k_idx = p2.cv_k_idx
                max_score = p2_score

        # If, score get higher score than thresh, terminate gradient searching
        if thresh_score != None and max_score > thresh_score: break



        # 3. Calcuate gradient of each theta(point).
        #    with using above theta value, set another theta(point).
        change_of_cost = p2_score - p1_score
        change_of_theta1 = p2.scalers_idx - p1.scalers_idx
        change_of_theta2 = p2.models_idx - p1.models_idx
        change_of_theta3 = p2.cv_k_idx - p1.cv_k_idx

        # If, attribute of theta1 and theta2 are same, set gradient value to 0 (slope = 0)
        def update_gradient_value(change_of_cost, change_of_theta):
            result_gradient = 0
            if change_of_theta != 0:
                result_gradient = change_of_cost / change_of_theta
            return result_gradient

        gradient_theta1 = update_gradient_value(change_of_cost, change_of_theta1)
        gradient_theta2 = update_gradient_value(change_of_cost, change_of_theta2)
        gradient_theta3 = update_gradient_value(change_of_cost, change_of_theta3)


        # 4. Prepare for next gradient (change theta 1 to new position)
        def set_new_point(gradient_theta, compare1, compare2):
            result_idx = 0
            if gradient_theta > 0:
                result_idx = max([compare1, compare2])
            elif gradient_theta < 0:
                result_idx = min([compare1, compare2])
            else: 
                result_idx = compare1
            return result_idx

        # Set new theta1 for the next calculation
        p1.scalers_idx = set_new_point(gradient_theta1, p1.scalers_idx, p2.scalers_idx)
        p1.models_idx = set_new_point(gradient_theta2, p1.models_idx, p2.models_idx)
        p1.cv_k_idx = set_new_point(gradient_theta3, p1.cv_k_idx, p2.cv_k_idx)


    class res:
        best_params = {}

    res.best_params = {
        'best_scaler': scalers[max_scalers_idx],
        'best_model' : models[max_models_idx],
        'best_cv_k': cv_k[max_cv_k_idx],
    }
    res.best_scaler = scalers[max_scalers_idx]
    res.best_model = models[max_models_idx]
    res.best_k = cv_k[max_cv_k_idx]

    res.best_score = max_score

    # Return value with dictionary type
    return res

  
  
  
  
def logo():
    print("")
    print("        /‾‾‾‾‾‾\      /‾‾‾\   /‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾\  /‾‾‾‾‾‾‾‾‾‾‾‾\       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\  /‾‾‾‾\     ")
    print("       /        \    /    /\ /    / /            /\/             /\     /                  /\/     /\    ")
    print("      /    /\    \  /    /  /    / /\‾‾/    /\‾‾\ /    /‾‾‾/    / /    /    /‾/    /‾/    / /     / /    ")
    print("     /    / /    / /    /  /    / /  ‾/    / /‾‾‾/    /   /    / /    /    / /    / /    / /     / /     ")
    print("    /     ‾‾    / /    /__/    / /   /    / /   /    /___/    / /    /    / /    / /    / /     / /      ")
    print("   /    /‾/    / /            / /   /    / /   /             / /    /    / /    / /    / /      ‾‾‾‾‾‾/\ ")
    print("  /____/ /____/ /____________/ /   /____/ /   /_____________/ /    /____/ /____/ /____/ /____________/ / ")
    print("  \____\ \____\/\____________\/    \____\/    \_____________\/     \____\ \____\ \____\/ \___________\/  ")
    print("                                                                    for Classifier / Version: 2021.11.10 ")
