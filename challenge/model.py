import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from challenge.preprocess_utils import *

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

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
        threshold_in_minutes = 15
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        if target_column: # If a target column is provided, return both features and target
            if target_column in data.columns: # Ensure the target column exists
                target = data[target_column]
                features = data.drop(target_column, axis=1)
                return features, target
            else:
                raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Return only features
        return data

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
        training_data = shuffle(features[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)

        features_df = pd.concat([
                pd.get_dummies(training_data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(training_data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(training_data['MES'], prefix = 'MES')], 
                axis = 1
            )
        model_target = target

        # Based on Data Analysis
        top_10_features = [
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

        # Data Scale
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0/n_y1
        
        #
        x_train, x_test, y_train, y_test = train_test_split(features_df[top_10_features], model_target, test_size = 0.33, random_state = 42)
        print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(x_train, y_train)

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

        if self._model:
            preds = self._model.predict(features)

            return preds
        else:
            raise Exception("Model not found.")