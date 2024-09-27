import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    TimeSeriesSplit, 
    GridSearchCV, 
    RandomizedSearchCV
)
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
#ot
tscv = TimeSeriesSplit(n_splits=5)

def weighted_mae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    """def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
    """

def evaluate_models(models, X_train: np.ndarray, y_train: np.ndarray):
    tscv = TimeSeriesSplit(n_splits=5)
    wmae_results = []
    best_model_name = None
    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        pipeline = Pipeline([('model', model)])
        
        train_errors = []
        val_errors = []
        
        for i, (train_index, val_index) in enumerate(tscv.split(X_train)):
            X_t, X_v = X_train[train_index], X_train[val_index]
            y_t, y_v = y_train[train_index], y_train[val_index]
            
            pipeline.fit(X_t, y_t)
            y_pred_train = pipeline.predict(X_t)
            y_pred_val = pipeline.predict(X_v)
            
            # calcular los pesos para WMAE
            weights_train = np.ones_like(y_t)
            weights_val = np.ones_like(y_v)
            
            train_wmae = weighted_mae(y_t, y_pred_train, weights_train)
            val_wmae = weighted_mae(y_v, y_pred_val, weights_val)
            
            train_errors.append(train_wmae)
            val_errors.append(val_wmae)
            #calculo la media de los errores 



            
            print(f'{name} - Split {i+1} - WMAE entrenamiento: {train_wmae:.4f}, WMAE validación: {val_wmae:.4f}')
        

        plt.figure(figsize=(10, 6))
        plt.title(f'Overfitting curve: {name}')
        plt.plot(range(1, tscv.n_splits + 1), train_errors, 'b-o', label='Entrenamiento')
        plt.plot(range(1, tscv.n_splits + 1), val_errors, 'r-o', label='Validación')
        plt.xlabel('Split')
        plt.ylabel('WMAE')
        plt.legend()
        plt.show()
        
        avg_val_wmae = np.mean(val_errors)
        #sacar la std
        wmae_results.append({'model': name, 'WMAE promedio': avg_val_wmae})

        
        if avg_val_wmae < best_score:
            best_score = avg_val_wmae
            best_model_name = name
            best_model = model
    
    result_df = pd.DataFrame(wmae_results)
    
    print(f'\nMejor modelo: {best_model_name} con WMAE promedio: {best_score:.4f}')
    
    return best_model, result_df


def plot_errors(results):
    model_names = list(results.keys())
    errors = [results[model]['score'] for model in model_names]

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, errors, color='skyblue')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.title('Model Comparison Based on Cross-Validated Error')
    plt.show()

