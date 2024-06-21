from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import pandas as pd
import numpy as np

def get_best_hyperparameters(train_x, train_y, val_x, val_y, parameters = None, criterion = 'accuracy'):
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf'],
    'degree': [3, 4, 5],
    'gamma': ['auto'] + [0.001, 0.01, 0.1, 1, 10]
    }
    # param_grid = {'scaler': StandardScaler(), 'svc': SVC(cache_size=2000),
    #                 'svc__C': [0.1, 1, 10, 100],
    #                 'svc__kernel': ['poly', 'rbf'],
    #                 'svc__degree': [3, 4, 5],
    #                 'svc__gamma': ['auto'] + [0.001, 0.01, 0.1, 1, 10]}
    
    if parameters is not None:
        param_grid = parameters
    
    split_index = [-1]*len(train_x) + [0]*len(val_x)
    X = np.concatenate((train_x, val_x), axis=0)
    y = np.concatenate((train_y, val_y), axis=0)
    pds = PredefinedSplit(test_fold = split_index)
    
    # model = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svc', SVC(cache_size=2000))
    # ])

    model = SVC(cache_size=2000)


    grid_search = GridSearchCV(model, param_grid = param_grid, cv = pds, scoring = criterion)
    grid_search.fit(X, y)  
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # print("Best Hyperparameters: ", best_params)
    # print("Best Score: ", best_model.score(val_x, val_y))
    return best_model 