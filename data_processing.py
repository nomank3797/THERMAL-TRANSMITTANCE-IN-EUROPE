import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import nan, isnan, array

# Function to normalize data using Min-Max Scaling
def normalize_data(values):
    """
    Normalize the input data to a range between 0 and 1.

    Parameters:
    - values: Array-like, the data to be normalized.

    Returns:
    - normalized: Array, the normalized data.
    """
    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler to the data
    scaler.fit(values)
    
    # Transform the data using the fitted scaler
    normalized = scaler.transform(values)
    
    return normalized

# Function to split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in=1, n_steps_out=1):
    """
    Split a multivariate sequence into input/output samples for supervised learning.

    Parameters:
    - sequences: Array-like, the sequence data to be split.
    - n_steps_in: Integer, the number of input time steps.
    - n_steps_out: Integer, the number of output time steps.

    Returns:
    - X: Array, the input samples.
    - y: Array, the output samples.
    """
    X, y = list(), list()
    
    # Iterate over the length of the sequence
    for i in range(len(sequences)):
        # Determine the end index of the input sequence
        end_ix = i + n_steps_in
        
        # Determine the end index of the output sequence
        out_end_ix = end_ix + n_steps_out
        
        # Check if the end index is beyond the length of the dataset
        if out_end_ix > len(sequences) + 1:
            break
        
        # Extract the input and output parts of the sequence
        seq_x = sequences[i:end_ix, :-2]
        seq_y = sequences[end_ix-1:out_end_ix-1, -2:]
        
        # Append the sequences to the respective lists
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)
