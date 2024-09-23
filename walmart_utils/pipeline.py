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
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ]
        )
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
    
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
    
    def get_final_columns(self) -> list[str]:
        """Return the final columns after transformation.

        Returns:
            list[str]: List of final column names after transformation.
        """
        # Obtén los nombres de las columnas transformadas
        transformed_num_cols = self.numerical_cols  # Columnas numéricas no cambian
        transformed_cat_cols = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_cols)

        # Combina ambas listas
        final_columns = np.concatenate([transformed_num_cols, transformed_cat_cols]).tolist()
        return final_columns