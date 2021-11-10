
from math import nan
from matplotlib.pyplot import sca
from pandas.core.frame import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import random





def brute_force(
    X:DataFrame,
    y:DataFrame,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"), 
        LogisticRegression(solver='lbfgs'), LogisticRegression(solver='newton-cg'), LogisticRegression(solver='liblinear'), LogisticRegression(solver='sag'), LogisticRegression(solver='saga'), 
        SVC(kernel='rbf',probability=True),SVC(kernel='rbf', gamma = 0.001,probability=True),SVC(kernel='rbf', gamma = 0.01,probability=True),SVC(kernel='rbf', gamma = 0.1,probability=True),SVC(kernel='rbf', gamma = 1,probability=True),SVC(kernel='rbf', gamma = 10,probability=True),
        SVC(kernel='poly',probability=True),SVC(kernel='poly', gamma = 0.001,probability=True),SVC(kernel='poly', gamma = 0.01,probability=True),SVC(kernel='poly', gamma = 0.1,probability=True),SVC(kernel='poly', gamma = 1,probability=True),SVC(kernel='poly', gamma = 10,probability=True),
        SVC(kernel='sigmoid',probability=True),SVC(kernel='sigmoid', gamma = 0.001,probability=True),SVC(kernel='sigmoid', gamma = 0.01,probability=True),SVC(kernel='sigmoid', gamma = 0.1,probability=True),SVC(kernel='sigmoid', gamma = 1,probability=True),SVC(kernel='sigmoid', gamma = 10,probability=True),
        SVC(kernel='linear',probability=True),SVC(kernel='linear', gamma = 0.001,probability=True),SVC(kernel='linear', gamma = 0.01,probability=True),SVC(kernel='linear', gamma = 0.1,probability=True),SVC(kernel='linear', gamma = 1,probability=True),SVC(kernel='linear', gamma = 10,probability=True),
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    is_cv_shuffle = True,
):
    """
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
      - DecisionTreeClassifier, LogisticRegression, SVC as default with hyperparameters.
    - `cv_k`: array
      - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].
    - `is_cv_shuffle`
      - To set shuffle or not in cross validation

    Returns
    ----------
    - `best_params_`: dictionary type of results.
    - `best_scaler_`: Scaler what has best score.
    - `best_model_`: Model what has best score.
    - `best_cv_k_`: k value in K-fold CV what has best score.
    - `best_score_`: double
      - Represent the score of the `best_params`.
    """

    print("Total trials: ", len(scalers) * len(models) * len(cv_k))

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
                score_result = cross_val_score(models[m], X, y, cv=kfold)
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
    max_iter = 100,
):
    """
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
      - DecisionTreeClassifier, LogisticRegression, SVC as default with hyperparameters.
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
    - `best_scaler_`: Scaler what has best score.
    - `best_model_`: Model what has best score.
    - `best_cv_k_`: k value in K-fold CV what has best score.
    - `best_score_`: double
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
            p1_X = scalers[scalers_idx].fit_transform(X)
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
    max_iter = 100,
):
    """
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
      - DecisionTreeClassifier, LogisticRegression, SVC as default with hyperparameters.
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
    - `best_scaler_`: Scaler what has best score.
    - `best_model_`: Model what has best score.
    - `best_cv_k_`: k value in K-fold CV what has best score.
    - `best_score_`: double
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
            # print("-------------")
            # print(point_val+1, end=" / ")
            # print(max_len)
            result = 0
            if target_gradient_theta > 0 and point_val+1 != max_len:
                # print("gradient ++")
                # if point_val+1 == max_len => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(point_val + 1, max_len)

            elif target_gradient_theta < 0 and point_val != 0:
                # print("gradient --")
                # if point_val == 0 => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(0, point_val)
            
            else:
                # print("gradient = 0")
                result = random.randrange(0, max_len)

            return result


        p2 = Point()
        p2.scalers_idx = check_gradient(gradient_theta1, p1.scalers_idx, scalers_len)
        p2.models_idx = check_gradient(gradient_theta2, p1.models_idx, models_len)
        p2.cv_k_idx = check_gradient(gradient_theta3, p1.cv_k_idx, cv_k_len)

        # print("First pick p1")
        # print(p1.scalers_idx)
        # print(p1.models_idx)
        # print(p1.cv_k_idx)

        # print("First pick p2")
        # print(p2.scalers_idx)
        # print(p2.models_idx)
        # print(p2.cv_k_idx)




        # 2. Calculate score(J(theta)) of each theta(point)
        p1_score = 0
        p2_score = 0

        # Check mem_table if score already has been calculated
        if mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx] != 0:
            p1_score = mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx]
        else:
            # if not, calculate score of theta
            p1_X = scalers[p1.scalers_idx].fit_transform(X)
            kfold = KFold(n_splits=cv_k[p1.cv_k_idx], shuffle=is_cv_shuffle)
            p1_score = cross_val_score(models[p1.models_idx], p1_X, y, cv=kfold).mean()
            # 2-1. Memoization
            mem_table[p1.scalers_idx][p1.models_idx][p1.cv_k_idx] = p1_score

        if mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx] != 0:
            p2_score = mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx]
        else:
            p2_X = scalers[p2.scalers_idx].fit_transform(X)
            kfold = KFold(n_splits=cv_k[p2.cv_k_idx], shuffle=is_cv_shuffle)
            p2_score = cross_val_score(models[p2.models_idx], p2_X, y, cv=kfold).mean()
            # 2-1. Memoization
            mem_table[p2.scalers_idx][p2.models_idx][p2.cv_k_idx] = p2_score    

        # print("Scores")
        # print(p1_score)
        # print(p2_score)

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





        # Calcuate gradient
        # 각 세타값에 대해 계산하고, 해당 기울기를 기준으로 다음 세타값을 잡음
        change_of_cost = p2_score - p1_score
        change_of_theta1 = p2.scalers_idx - p1.scalers_idx
        change_of_theta2 = p2.models_idx - p1.models_idx
        change_of_theta3 = p2.cv_k_idx - p1.cv_k_idx

        # 이전 지점과 같을 때, 0으로 처리
        def update_gradient_value(change_of_cost, change_of_theta):
            result_gradient = 0
            if change_of_theta != 0:
                result_gradient = change_of_cost / change_of_theta
            return result_gradient

        gradient_theta1 = update_gradient_value(change_of_cost, change_of_theta1)
        gradient_theta2 = update_gradient_value(change_of_cost, change_of_theta2)
        gradient_theta3 = update_gradient_value(change_of_cost, change_of_theta3)

        # print("Gradient")
        # print(gradient_theta1)
        # print(gradient_theta2)
        # print(gradient_theta3)


        # Prepare for next gradient (change theta 1 to new position)

        def set_new_point(gradient_theta, compare1, compare2):
            result_idx = 0
            if gradient_theta > 0:
                result_idx = max([compare1, compare2])
            elif gradient_theta < 0:
                result_idx = min([compare1, compare2])
            else: 
                result_idx = compare1
            return result_idx



        p1.scalers_idx = set_new_point(gradient_theta1, p1.scalers_idx, p2.scalers_idx)
        p1.models_idx = set_new_point(gradient_theta2, p1.models_idx, p2.models_idx)
        p1.cv_k_idx = set_new_point(gradient_theta3, p1.cv_k_idx, p2.cv_k_idx)


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














def logo():
    print("")
    print("        /‾‾‾‾‾‾‾\    /‾‾‾‾/  /‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾‾‾/\     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾/\/‾‾‾‾‾/\   ")
    print("       /         \  /    /  /    / /            / /             / /    /                  / /     / /   ")
    print("      /    /\    /\/    /  /    / /\‾‾/    /\‾‾\ /    /‾‾‾/    / /    /    /‾/    /‾/    / /     / /    ")
    print("     /    / /   / /    /  /    / /  ‾/    / /‾‾‾/    /   /    / /    /    / /    / /    / /     / /     ")
    print("    /     ‾‾   / /    /__/    / /   /    / /   /    /___/    / /    /    / /    / /    / /     /_/____  ")
    print("   /    /‾/   / /            / /   /    / /   /             / /    /    / /    / /    / /            /\ ")
    print("  /____/ /___/ /____________/ /   /____/ /   /_____________/ /    /____/ /____/ /____/ /____________/ / ")
    print("  \____\ \___\/\____________\/    \____\/    \_____________\/     \____\ \____\ \____\/ \___________\/  ")
    print("                                                                                    Version: 2021.11.10 ")