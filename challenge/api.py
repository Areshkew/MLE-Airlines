import fastapi
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel
from fastapi import HTTPException

app = fastapi.FastAPI()
model = None 
valid_values = {}

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictionInput(BaseModel):
    flights: list[Flight]
    
@app.on_event("startup")
def load_model():
    global model
    
    # 
    data_path = "data/data.csv"
    data = pd.read_csv(data_path)
    
    # Store the unique values for OPERA, TIPOVUELO, and MES
    valid_values['OPERA'] = set(data['OPERA'].unique())
    valid_values['TIPOVUELO'] = set(data['TIPOVUELO'].unique())
    valid_values['MES'] = set(data['MES'].unique())
    
    #
    model = DelayModel()

    #
    features, target = model.preprocess(
        data = data,
        target_column="delay"
    )

    model.fit(
        features=features,
        target=target
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(input_data: PredictionInput) -> dict:
    global model

    if model is None:
        raise HTTPException(status_code=400, detail="Model is not initialized.")

    # 
    input_df = pd.DataFrame([flight.dict(by_alias=True) for flight in input_data.flights])

    for _, row in input_df.iterrows():
        if row['OPERA'] not in valid_values['OPERA']:
            raise HTTPException(status_code=400, detail=f"OPERA value '{row['OPERA']}' is not valid.")
        if row['TIPOVUELO'] not in valid_values['TIPOVUELO']:
            raise HTTPException(status_code=400, detail=f"TIPOVUELO value '{row['TIPOVUELO']}' is not valid.")
        if row['MES'] not in valid_values['MES']:
            raise HTTPException(status_code=400, detail=f"MES value '{row['MES']}' is not valid.")

    # 
    try:
        processed_input = model.preprocess(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing error: {e}")

    #
    try:
        prediction = model.predict(processed_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return {"predict": prediction}