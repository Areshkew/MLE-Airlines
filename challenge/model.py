import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Optional, Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from challenge.preprocess_utils import *

class DelayModel:

    def __init__(
        self
    ):
        params = {
            'random_state': 1,
            'learning_rate': 0.01,
            'scale_pos_weight': 4.40 # Acordding to the Third Sight the aprox scale of our model is 4.40
        }

        self._model = xgb.XGBClassifier( **params )

        self.FEATURES_COLS = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Feature Generation
        if 'Fecha-I' in data.columns:
            threshold_in_minutes = 15
            data['period_day'] = data['Fecha-I'].apply(get_period_day)
            data['high_season'] = data['Fecha-I'].apply(is_high_season)
            data['min_diff'] = data.apply(get_min_diff, axis = 1)
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix = 'MES')], 
                axis = 1
        )

        # Fill Missing Values
        default_value = 0
        missing_columns = [col for col in self.FEATURES_COLS if col not in features.columns]
        for col in missing_columns:
            features[col] = default_value

        if target_column: # If a target column is provided, return both features and target
            if target_column in data.columns: # Ensure the target column exists
                target = data[['delay']]
                
                return features[self.FEATURES_COLS], target
            else:
                raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Return only features
        return features[self.FEATURES_COLS]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        preds = self._model.predict(features)

        return preds.tolist()
