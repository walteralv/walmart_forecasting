from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

models = {
    'randomforest': {
        'model': RandomForestRegressor,
        'params': {
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'n_estimators': [20, 50, 100, 150, 200, 250, 300,500],
            'min_samples_split': [2, 3, 4, 5, 10]
        }
    },
    'xgboost': {
        'model': XGBRegressor,
        'params': {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'n_estimators': [30, 50, 100, 150, 200, 250, 300, 500],
            'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001]
        }
    },
    'lightgbm': {
        'model': LGBMRegressor,
        'params': {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'n_estimators': [30, 50, 100, 150, 200, 250, 300, 500],
            'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001]
        }
    }, 
    'linearregression': {
        'model': LinearRegression,
        'params': {

        }  
    },
    'ridge': {
        'model': Ridge,
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0],  
            'max_iter': [1000, 2000, 3000],    
            'solver': ['auto', 'sag', 'saga']   
        }
    },
    'knn': {
        'model': KNeighborsRegressor,
        'params': {
            'n_neighbors': [3, 5, 7, 10, 15],  
            'weights': ['uniform', 'distance'],  # weight function
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # algorithm to compute nearest neighbors
        }
    }

}