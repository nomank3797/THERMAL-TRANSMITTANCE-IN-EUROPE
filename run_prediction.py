import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import custom modules
import data_processing 
import models 

# Define function to process files within an RCP folder
def process_rcp_folder(rcp_folder_path):
    # Initialize lists to hold combined training and testing data
    X_train_combined = []
    y_train_combined = []
    X_test_combined = []
    y_test_combined = []

    # Iterate over each file in the RCP folder
    for file_name in os.listdir(rcp_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(rcp_folder_path, file_name)
            print(f"Processing file: {file_path}")

            # Load CSV data
            dataset = pd.read_csv(file_path, header=0)
            
            # Set the number of time steps for input and output
            n_steps_in, n_steps_out = 1, 1

            # Define the number of epochs for training
            epochs = 10

            # Convert data into input/output sequences
            X, y = data_processing.split_sequences(dataset.values, n_steps_in, n_steps_out)
            
            # Reshape data to [rows, columns] structure
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

            # Normalize the data
            normalized_X = data_processing.normalize_data(X)
            normalized_y = data_processing.normalize_data(y)
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(normalized_X, normalized_y, test_size=0.2)

            # Append the split data to the combined lists
            X_train_combined.append(X_train)
            y_train_combined.append(y_train)
            X_test_combined.append(X_test)
            y_test_combined.append(y_test)

    # Combine all training and testing data
    X_train_combined = np.vstack(X_train_combined)
    y_train_combined = np.vstack(y_train_combined)
    X_test_combined = np.vstack(X_test_combined)
    y_test_combined = np.vstack(y_test_combined)

    # Train the model and evaluate its performance
    mse, mae, rmse = models.lstm_model(X_train_combined, X_test_combined, y_train_combined, y_test_combined, n_steps_in, n_steps_out, epochs)
    return mse, mae, rmse

# Define function to process data for each country
def process_country_data(country_folder):
    results = []

    # Iterate over each city folder within the country folder
    for city_folder in os.listdir(country_folder):
        city_path = os.path.join(country_folder, city_folder)
        
        # Process each RCP scenario folder within the city folder
        for rcp_folder in ['RCP2.6', 'RCP4.5', 'RCP8.5']:
            rcp_path = os.path.join(city_path, rcp_folder)
            print(f"Processing RCP folder: {rcp_path}")

            # Process the RCP folder and collect results
            mse, mae, rmse = process_rcp_folder(rcp_path)

            # Append results if evaluation metrics are valid
            if mse is not None:
                results.append({
                    'Country': os.path.basename(country_folder),
                    'City': city_folder,
                    'RCP': rcp_folder,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                })

    return results

# Main function to process all data and save results
def main(data_folder, results_file):
    all_results = []

    # Iterate over each country folder within the data folder
    for country_folder in os.listdir(data_folder):
        country_path = os.path.join(data_folder, country_folder)
        if os.path.isdir(country_path):
            country_results = process_country_data(country_path)
            all_results.extend(country_results)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {results_file}")

# Input data folder path
data_folder = 'Clean Data'  
# Results file path
results_file = 'results.csv'
# Call main function to process data and save results
main(data_folder, results_file)
