
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





def auto_ml():
    print("auto ml")