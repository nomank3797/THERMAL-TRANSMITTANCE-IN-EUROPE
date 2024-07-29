# Importing necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from keras.optimizers import Adam
import pandas as pd

# Import custom module for forecast evaluation
import forecast_evaluation  

# Define the LSTM model function
def lstm_model(x_train, x_test, y_train, y_test, n_steps_in, n_steps_out, epochs=10):
    """
    Builds, trains, and evaluates an LSTM model for time series forecasting.

    Parameters:
    - x_train: Training input data
    - x_test: Testing input data
    - y_train: Training target data
    - y_test: Testing target data
    - n_steps_in: Number of input time steps
    - n_steps_out: Number of output time steps
    - epochs: Number of training epochs (default is 10)

    Returns:
    - mse: Mean Squared Error of the model predictions
    - mae: Mean Absolute Error of the model predictions
    - rmse: Root Mean Squared Error of the model predictions
    """

    # Reshaping input data to the required 3D shape for LSTM
    x_train = x_train.reshape(x_train.shape[0], n_steps_in, x_train.shape[1])
    y_train = y_train.reshape(y_train.shape[0], n_steps_out, y_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], n_steps_in, x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0], n_steps_out, y_test.shape[1])

    # Determine the number of features from the reshaped input data
    n_features = x_train.shape[2]
      
    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(200, activation='relu', kernel_initializer='he_uniform', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', kernel_initializer='he_uniform', return_sequences=True))
    model.add(TimeDistributed(Dense(y_test.shape[2], activation='linear')))

    # Compiling the model with Adam optimizer and Mean Squared Error loss function
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse')

    # Training the model with the training data
    model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=2)

    # Making predictions with the trained model on the test data
    yhat = model.predict(x_test, verbose=2)

    # Reshaping the predictions and test data for evaluation
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1] * yhat.shape[2])
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2])

    # Evaluating the model predictions using custom evaluation function
    mse, mae, rmse = forecast_evaluation.evaluate_forecasts(y_test, yhat)

    return mse, mae, rmse
