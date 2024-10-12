import unittest
from unittest.mock import ANY

from tests.custom_testclient import TestClient
from mockito import when
import numpy as np
from challenge.api import app

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_should_get_predict(self):
        with self.client as client:
            data = {
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "N", 
                        "MES": 3
                    }
                ]
            }
            when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unkown_column_1(self):
        with self.client as client:
            data = {       
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "N",
                        "MES": 13
                    }
                ]
            }
            when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        with self.client as client:
            data = {        
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "O", 
                        "MES": 13
                    }
                ]
            }
            when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)
    
    def test_should_failed_unkown_column_3(self):
        with self.client as client:
            data = {        
                "flights": [
                    {
                        "OPERA": "Argentinas", 
                        "TIPOVUELO": "O", 
                        "MES": 13
                    }
                ]
            }
            when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)