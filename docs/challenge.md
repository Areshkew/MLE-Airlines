<!-- Documentation and Explanations -->

# Part I
- Model Choosing: XgBoost
    I am choosing XgBoost because it handles large datasets and complex relationships between features and we don't need standardization
    like LinearRegression where we need to scale our data so that they have common characteristics and it has Built-in regularization to prevent overfitting.

    For the improved model I chose **XGBoost with Feature Importance and with Balance** because the balancing of class weights further ensures that the model pays equal attention to both classes, reducing the risk of bias towards the majority class and enhancing the detection of the minority class.

    A balance betwen precision and recall is desirable, and this is often captured using metrics like the F1 score.