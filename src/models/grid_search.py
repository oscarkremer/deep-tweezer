from sklearn.model_selection import GridSearchCV
from src.utils import suppress_stdout_stderr

def grid_search(X_train, X_test, y_train, model):
    """[Summary]

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    parameters = model.params
    gb_clf = model.model
    with suppress_stdout_stderr():
        clf = GridSearchCV(gb_clf, parameters, scoring="roc_auc", cv=5, verbose=0, n_jobs=6)
        clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:,1]
    return y_score