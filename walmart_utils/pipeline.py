from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class GlobalPipeline:
    def __init__(
            self, numerical_cols: list[str], categorical_cols: list[str]     
        ) -> None:

        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            #('imputer', SimpleImputer(strategy='most_frequent')),  
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ]
        )
    
    def fit(self, data: pd.DataFrame | np.ndarray) -> None:
        """learn the transformations
        Args:
            data (pd.DataFrame | np.ndarray): _description_
        """
        
        self.preprocessor.fit(data)

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """apply the learned transformations to the data

        Args:
            data (pd.DataFrame | np.ndarray): _description_

        Returns:
            np.ndarray: transformed data
        """
        return self.preprocessor.transform(data)
    

    def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """learn the transformations and apply the learned transformations to the data

        Args:
            data (pd.DataFrame | np.ndarray): _description_

        Returns:
            np.ndarray:transformed data
        """
        return self.preprocessor.fit_transform(data)
    
    def get_preprocessor(self) -> ColumnTransformer:
        """Return the internal preprocessor object

        Returns:
            ColumnTransformer: ColumnTransformer
        """
        return self.preprocessor