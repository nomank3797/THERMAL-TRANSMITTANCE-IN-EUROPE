from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt

# Function to evaluate one or more forecasts against expected values
def evaluate_forecasts(actual, predicted):
    """
    Evaluate one or more forecasts against the expected values.

    Parameters:
    - actual (array-like): Actual values.
    - predicted (array-like): Predicted values.

    Returns:
    - mse: Mean Squared Error between actual and predicted values.
    - mae: Mean Absolute Error between actual and predicted values.
    - rmse: Root Mean Squared Error between actual and predicted values.
    """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual, predicted)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = sqrt(mse)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)

    # Print evaluation metrics
    print("MSE: {:.4f}".format(mse))
    print("RMSE: {:.4f}".format(rmse))
    print("MAE: {:.4f}".format(mae))
    
    return mse, mae, rmse
