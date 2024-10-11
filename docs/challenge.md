<!-- Documentation and Explanations -->
# Requirements
 - No dependencies were originally changed. 
 - Python version used: 3.10.14

# Part I
- Model Choosing: XgBoost
    I am choosing XgBoost because it handles large datasets and complex relationships between features and we don't need standardization
    like LinearRegression where we need to scale our data so that they have common characteristics and it has Built-in regularization to prevent overfitting.

    For the improved model I chose **XGBoost with Feature Importance and with Balance** because the balancing of class weights further ensures that the model pays equal attention to both classes, reducing the risk of bias towards the majority class and enhancing the detection of the minority class.

    A balance betwen precision and recall is desirable, and this is often captured using metrics like the F1 score.

- The model hyperparameters were adjusted according to our data analysis, in this case we used a scale_pos_weight of 4.40 aproximately.


# Part II
- I decided to train and load the model during app startup because it ensures that the model is ready for use as soon as the application is launched instead of using a model file for simplicity and to avoid the overhead of uploading files. This approach streamlines the process, as it eliminates the need to manage and store separate model files, reducing potential points of failure.

- Based on tests the input of our '/predict' endpoint needs to have this structure always:
    ```python
    class Flight(BaseModel):
        OPERA: str
        TIPOVUELO: str
        MES: int


    class PredictionInput(BaseModel):
        flights: list[Flight]
    ```

    that translates to:
    ```
    {
        "flights": [
            {
            "OPERA": "Aerolineas Argentinas",
            "TIPOVUELO": "N",
            "MES": 3
            }
        ]
    }
    ```
- Found a FastApi bug in my specific version of python when testing module `'anyio' has no attribute 'start_blocking_portal`.
    For fixing it an running the test properly I referenced the `start_blocking_portal` function in anyio.from_thread package in stable version.

    Also as I loaded my model on api startup, I needed to configure the tests properly according to: 
        [Advanced | Testing Events (FastApi)](https://fastapi.tiangolo.com/advanced/testing-events/)