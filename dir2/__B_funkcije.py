# from __A_knjiznice import *

from PyFingerprint.fingerprint import get_fingerprint

import pandas as pd
import numpy as np
from PyFingerprint.fingerprint import get_fingerprint

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

############
def get_fingerprint_wrapper(args):
    from PyFingerprint.fingerprint import get_fingerprint
    smiles, FINGERPRINT = args
    output = get_fingerprint(smiles, FINGERPRINT)
    finger_np = np.array([output.to_numpy()])
    return finger_np

def calc_fingerprints(X, FINGERPRINT):
    # Assuming you have a pandas DataFrame named X with a column named "Smiles"
    smiles_column = X["Smiles"].tolist()  # Convert to list for multiprocessing
    num_chunks = 12
    num_smiles = len(smiles_column)

    # Check if there are any SMILES strings to process
    if num_smiles == 0:
        raise ValueError("The input DataFrame has no SMILES strings to process.")

    # Calculate chunk size
    chunk_size = max(1, num_smiles // num_chunks)  # Ensure chunk_size is at least 1

    # Create chunks of SMILES strings
    chunks = [smiles_column[i:i + chunk_size] for i in range(0, num_smiles, chunk_size)]

    # Prepare arguments for multiprocessing
    args = [(smiles, FINGERPRINT) for chunk in chunks for smiles in chunk]

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(get_fingerprint_wrapper, args)

    # Convert results to DataFrame
    bit_columns_df = []
    for finger_np in results:
        num_bits = finger_np.shape[1]
        finger_df = pd.DataFrame(finger_np, columns=[f"Bit_{i}" for i in range(num_bits)])
        bit_columns_df.append(finger_df)

    # Concatenate the fingerprints DataFrames
    fingerprints_df = pd.concat(bit_columns_df, axis=0, ignore_index=True)
    fingerprints_df.index = X.index

    # Concatenate the original DataFrame X with the fingerprints DataFrame
    df = pd.concat([X, fingerprints_df], axis=1)

    return df, fingerprints_df

import modin.pandas as pdm

import pandas as pd  # Ensure pandas is explicitly imported
import modin.pandas as pdm
import numpy as np
import time
from multiprocessing import Pool


def calc_fingerprint_new(X, FINGERPRINT):
    """Calculate fingerprints for SMILES strings in parallel."""
    
    # Ensure X is converted to pandas DataFrame
    if isinstance(X, pdm.DataFrame):
        X = X._to_pandas()
    
    smiles_column = X["Smiles"].tolist()
    num_smiles = len(smiles_column)

    if num_smiles == 0:
        raise ValueError("The input DataFrame has no SMILES strings to process.")

    args = [(smiles, FINGERPRINT) for smiles in smiles_column]

    print("Starting fingerprint calculations...")
    start_time = time.time()

    # Parallel fingerprint calculation
    with Pool(processes=12) as pool:
        results = pool.map(get_fingerprint_wrapper, args)

    print(f"Fingerprint calculations completed in {time.time() - start_time:.2f} seconds.")

    # Preallocate a NumPy array for fingerprints
    num_bits = results[0].shape[1]  # Assuming all results have the same shape
    fingerprints_array = np.zeros((num_smiles, num_bits))  # Adjust the number of columns as needed

    # Fill the preallocated array with results
    for i, finger_np in enumerate(results):
        fingerprints_array[i] = finger_np.flatten()  # Flatten if necessary

    # Create the fingerprints DataFrame directly using pandas
    fingerprints_df = pd.DataFrame(fingerprints_array, columns=[f"Bit_{i}" for i in range(num_bits)])

    print("Starting concatenation of DataFrames...")
    concat_start_time = time.time()

    # Concatenate the original DataFrame X with the fingerprints DataFrame using pandas
    df = pdm.concat([X, fingerprints_df], axis=1)

    print(f"Concatenation completed in {time.time() - concat_start_time:.2f} seconds.")
    print("Final DataFrame shape:", df.shape)

    # Convert back to Modin DataFrame if necessary
    df = pdm.DataFrame(df)

    return df, pdm.DataFrame(fingerprints_df)



import modin.pandas as pdm  # Import Modin for parallel DataFrame operations
import numpy as np
from multiprocessing import Pool
import time

# def calc_fingerprint_new(X, FINGERPRINT, batch_size=500000):
#     """Calculate fingerprints for SMILES strings in parallel in batches."""
    
#     # Ensure X is already a Modin DataFrame (no need to convert if it's already in Modin format)
#     if isinstance(X, pdm.DataFrame):
#         X = X._to_pandas()  # Optional: In case X is a Modin DataFrame, convert to pandas (if needed).
    
#     # The process for SMILES extraction remains the same
#     smiles_column = X["Smiles"].tolist()
#     num_smiles = len(smiles_column)

#     if num_smiles == 0:
#         raise ValueError("The input DataFrame has no SMILES strings to process.")

#     # Initialize empty lists for final results
#     all_fingerprints = []
#     all_smiles = []

#     # Process in batches
#     for start_idx in range(0, num_smiles, batch_size):
#         end_idx = min(start_idx + batch_size, num_smiles)
#         batch_smiles = smiles_column[start_idx:end_idx]
        
#         # Prepare arguments for parallel processing
#         args = [(smiles, FINGERPRINT) for smiles in batch_smiles]

#         print(f"Starting fingerprint calculations for batch {start_idx} to {end_idx}...")
#         start_time = time.time()

#         # Parallel fingerprint calculation
#         with Pool(processes=12) as pool:
#             results = pool.map(get_fingerprint_wrapper, args)

#         print(f"Fingerprint calculations for batch {start_idx} to {end_idx} completed in {time.time() - start_time:.2f} seconds.")
        
#         # Collect results
#         all_fingerprints.extend(results)
#         all_smiles.extend(batch_smiles)

#     # Combine results into a single DataFrame
#     print("Starting concatenation of DataFrames...")
#     concat_start_time = time.time()

#     # Convert all fingerprints to a numpy array
#     num_bits = all_fingerprints[0].shape[1]  # Assuming all results have the same shape
#     fingerprints_array = np.zeros((len(all_fingerprints), num_bits))

#     for i, finger_np in enumerate(all_fingerprints):
#         fingerprints_array[i] = finger_np.flatten()  # Flatten if necessary

#     # Create the fingerprints DataFrame using Modin (pdm)
#     fingerprints_df = pdm.DataFrame(fingerprints_array, columns=[f"Bit_{i}" for i in range(num_bits)])

#     # Add SMILES and concatenate with the original DataFrame
#     smiles_df = pdm.DataFrame({'Smiles': all_smiles})
#     df = pdm.concat([smiles_df, fingerprints_df], axis=1)

#     print(f"Concatenation completed in {time.time() - concat_start_time:.2f} seconds.")
#     print("Final DataFrame shape:", df.shape)

#     return df, pdm.DataFrame(fingerprints_df)



###########
import pandas as pd
import modin.pandas as mpd
from rdkit import Chem
from multiprocessing import Pool


def is_valid_smiles(smi):
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except Exception:
        return False


def validate_smiles_wrapper(smiles):
    """Wrapper function for validating SMILES strings."""
    return is_valid_smiles(smiles)


def drop_invalid_smiles(X):
    """Drop invalid SMILES strings from the DataFrame using multiprocessing."""
    # Ensure we convert to pandas explicitly for compatibility
    if isinstance(X, mpd.DataFrame):
        X = X._to_pandas()
    
    # Extract SMILES column and validate in parallel
    smiles_column = X["Smiles"].tolist()
    
    print("Validating SMILES strings using multiprocessing...")
    with Pool(processes=12) as pool:  # Adjust the number of processes as needed
        valid_flags = pool.map(validate_smiles_wrapper, smiles_column)
    
    print(f"Found {sum(valid_flags)} valid SMILES out of {len(smiles_column)} total SMILES.")
    
    # Ensure valid_flags length matches the DataFrame's index
    if len(valid_flags) != len(X):
        raise ValueError("Mismatch between SMILES validation results and DataFrame length.")
    
    # Filter the DataFrame to keep only valid SMILES
    X_valid = X.loc[valid_flags].reset_index(drop=True)

    # Convert back to Modin DataFrame if needed
    return mpd.DataFrame(X_valid)


######### 

# batch_predictions.py

import numpy as np
from multiprocessing import Pool

def make_predictions_wrapper(model_data):
    """Wrapper function to make predictions for a chunk of data."""
    model, data_chunk = model_data
    return model.predict(data_chunk), model.predict_proba(data_chunk)

def make_predictions_in_batches(model, data, num_chunks=12):
    """
    Make predictions on data in batches using multiprocessing.

    Parameters:
    - model: The trained model to use for predictions.
    - data: The input data for which predictions are to be made.
    - num_chunks: The number of chunks to split the data into for parallel processing.

    Returns:
    - predictions: The predicted classes.
    - probabilities: The predicted probabilities for each class.
    """
    num_samples = data.shape[0]
    predictions = []
    probabilities = []

    # Calculate chunk size
    chunk_size = max(1, num_samples // num_chunks)

    # Create chunks of data
    chunks = [data[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]

    # Prepare arguments for multiprocessing
    args = [(model, chunk) for chunk in chunks]

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(make_predictions_wrapper, args)

    # Unzip the results
    for pred, proba in results:
        predictions.append(pred)
        probabilities.append(proba)

    # Concatenate results
    return np.concatenate(predictions), np.concatenate(probabilities)



# import os
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor

# def analyze_file(input_directory, filename):
#     file_path = os.path.join(input_directory, filename)  # Full path to the file
#     df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    
#     # Skip the first three columns
#     df_to_analyze = df.iloc[:, 3:]  # Select all columns except the first three
    
#     # Get the number of columns (features)
#     num_columns = df_to_analyze.shape[1]
    
#     # Check how many columns have varying values
#     varying_columns_count = df_to_analyze.nunique().gt(1).sum()  # Count columns with more than 1 unique value
    
#     # Initialize counts for same values in columns
#     same_values_100 = 0
#     same_values_95 = 0
    
#     # Analyze each column for same values
#     for column in df_to_analyze.columns:
#         unique_values = df_to_analyze[column].nunique()
        
#         # Check for 100% same values (all values are the same)
#         if unique_values == 1:
#             same_values_100 += 1
        
#         # Check for 95% same values (at least 95% of values are the same)
#         if (df_to_analyze[column].value_counts(normalize=True).max() >= 0.95):
#             same_values_95 += 1
            
#     # Calculate the share of columns with the same data
#     share_of_columns_with_same_data = same_values_100 / num_columns if num_columns > 0 else 0

#     return {
#         'Fingerprint': filename,  # Use the filename as the fingerprint name
#         'Total Features': num_columns,
#         'Varying Columns': varying_columns_count,
#         '100% same values in column': same_values_100,
#         '95% same values in column': same_values_95,
#         'Share of Columns with Same Data': share_of_columns_with_same_data,  # Add the new metric
#     }

# def analyze_fingerprints(input_directory):
#     # Create a list to hold the results
#     results = []

#     # Use ProcessPoolExecutor to analyze files in parallel
#     with ProcessPoolExecutor() as executor:
#         # Prepare arguments for each file
#         futures = {executor.submit(analyze_file, input_directory, filename): filename for filename in os.listdir(input_directory) if filename.endswith('.csv')}
        
#         # Collect results as they complete
#         for future in futures:
#             result = future.result()  # Get the result from the future
#             results.append(result)

#     # Create a DataFrame from the results
#     results_df = pd.DataFrame(results)

#     # Reorder the columns as specified
#     results_df = results_df[['Fingerprint', 'Total Features', 'Varying Columns', '100% same values in column', '95% same values in column', 'Share of Columns with Same Data']]

#     return results_df  # Return the results DataFrame

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from concurrent.futures import ProcessPoolExecutor

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def count_same_value_columns(x, thresholds=[1.0, 0.95, 0.9, 0.75]):
    '''
    Count columns with a high percentage of same values based on specified thresholds (1.0, 0.95, 0.9, 0.75).
    
    Parameters:
    - x: DataFrame with features.
    - thresholds: List of thresholds for same values percentage (e.g., 100%, 95%, 90%, 75%).
    
    Returns:
    - Dictionary with counts of columns for each threshold.
    '''
    same_value_counts = {threshold: 0 for threshold in thresholds}

    # Analyze each column for same values (i.e., only 0 or 1)
    for column in x.columns:
        value_counts = x[column].value_counts(normalize=True)
        max_percentage = value_counts.max()  # Find the highest percentage of a single value (either 0 or 1)

        # Check against each threshold
        for threshold in thresholds:
            if max_percentage >= threshold:
                same_value_counts[threshold] += 1

    return same_value_counts

def analyze_file(input_directory, filename):
    file_path = os.path.join(input_directory, filename)  # Full path to the file
    df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    
    # Skip the first three columns (assuming they are not features)
    df_to_analyze = df.iloc[:, 3:]  # Select all columns except the first three
    
    # Get the number of columns (features)
    num_columns = df_to_analyze.shape[1]
    
    # Count the number of columns with varying values
    varying_columns_count = df_to_analyze.nunique().gt(1).sum()  # Count columns with more than 1 unique value
    
    # Calculate the number of columns with 100%, 95%, 90%, and 75% same values
    same_value_counts = count_same_value_columns(df_to_analyze, thresholds=[1.0, 0.95, 0.9, 0.75])
    
    # Calculate the share of columns with the same data for each threshold
    share_of_columns_with_same_data = {threshold: same_value_counts[threshold] / num_columns if num_columns > 0 else 0 
                                       for threshold in same_value_counts}

    return {
        'Fingerprint': filename,  # Use the filename as the fingerprint name
        'Total Features': num_columns,
        'Varying Columns': varying_columns_count,
        '100% same values in column': same_value_counts[1.0],
        '95% same values in column': same_value_counts[0.95],
        '90% same values in column': same_value_counts[0.9],
        '75% same values in column': same_value_counts[0.75],
        'Share of 100% same values': share_of_columns_with_same_data[1.0],
        'Share of 95% same values': share_of_columns_with_same_data[0.95],
        'Share of 90% same values': share_of_columns_with_same_data[0.9],
        'Share of 75% same values': share_of_columns_with_same_data[0.75],
    }

def analyze_fingerprints(input_directory):
    # Create a list to hold the results
    results = []

    # Use ProcessPoolExecutor to analyze files in parallel
    with ProcessPoolExecutor() as executor:
        # Prepare arguments for each file
        futures = {executor.submit(analyze_file, input_directory, filename): filename for filename in os.listdir(input_directory) if filename.endswith('.csv')}
        
        # Collect results as they complete
        for future in futures:
            result = future.result()  # Get the result from the future
            results.append(result)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Reorder the columns as specified
    results_df = results_df[['Fingerprint', 'Total Features', 'Varying Columns',
                             '100% same values in column', '95% same values in column', 
                             '90% same values in column', '75% same values in column', 
                             'Share of 100% same values', 'Share of 95% same values',
                             'Share of 90% same values', 'Share of 75% same values']]

    return results_df  # Return the results DataFrame



##########




def remove_collinear_features_simple(x, threshold):
    '''
    Simplified function to remove collinear features in a DataFrame based on a specified correlation threshold.

    Parameters:
    - x: DataFrame with features.
    - threshold: Correlation coefficient threshold; features with correlations above this threshold will be removed.

    Returns:
    - DataFrame with collinear features removed.
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr().abs()

    # Identify pairs of columns that exceed the correlation threshold
    high_corr_pairs = [(i, j) for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns)) if corr_matrix.iloc[i, j] > threshold]

    # Create a set to hold all columns that need to be removed
    cols_to_remove = set()

    # Add one column from each pair to the set of columns to be removed
    for i, j in high_corr_pairs:
        cols_to_remove.add(corr_matrix.columns[j])

    # Remove the columns from the DataFrame
    x_reduced = x.drop(columns=list(cols_to_remove))

    return x_reduced

############
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from joblib import Parallel, delayed  # <-- Add this import to fix the error

# Ensure TensorFlow uses limited number of threads
def configure_tensorflow():
    tf.config.threading.set_intra_op_parallelism_threads(6)  # Limit to 6 threads for operations
    tf.config.threading.set_inter_op_parallelism_threads(6)  # Limit to 6 threads for parallelism between operations

# Define the neural network model
def create_nn_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def run_training(input_directory):
    configure_tensorflow()  # Set TensorFlow to use limited threads

    classifiers = [
        ('rf', RandomForestClassifier(n_jobs=6)),  # Set to 6 threads for RandomForest
        ('et', ExtraTreesClassifier(n_jobs=6)),    # Set to 6 threads for ExtraTrees
        ('xgb', XGBClassifier(eval_metric='logloss', n_jobs=6)),  # Set to 6 threads for XGBoost
        ('nn', lambda input_dim: KerasClassifier(build_fn=lambda: create_nn_model(input_dim), epochs=10, batch_size=32, verbose=0))
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'}
    results_list = []

    fingerprint_dfs = []
    filenames = []

    # Load the CSV files
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_directory, filename))
            fingerprint_dfs.append(df)
            filenames.append(filename)

    # Parallelize the processing of fingerprint data
    def process_fingerprint(idx, df, filename):
        print(f'Processing fingerprint DataFrame {idx + 1}/{len(fingerprint_dfs)}')

        fingerprint_name = filename.split('df_')[1].split('.')[0]
        y = df[['Activity']].values.ravel()
        X = df.iloc[:, 3:]
        input_dim = X.shape[1]

        X_interim, X_test, y_interim, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_interim, y_interim, test_size=15/85, random_state=42, stratify=y_interim)

        for clf_name, clf_factory in classifiers:
            print(f"Training {clf_name}...")

            # Initialize the classifier
            if clf_name == 'nn':
                classifier = clf_factory(input_dim)
            else:
                classifier = clf_factory

            # Cross-validation
            cv_results = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring, n_jobs=6)  # Use 6 parallel jobs

            # Fit the model
            classifier.fit(X_train, y_train)
            y_val_pred = classifier.predict(X_val)
            y_val_pred_binary = (y_val_pred > 0.5) if clf_name == 'nn' else y_val_pred

            results_list.append({
                'Fingerprint': fingerprint_name,
                'Classifier': clf_name,
                'CV_Mean_Accuracy': cv_results['test_accuracy'].mean(),
                'CV_Mean_F1': cv_results['test_f1'].mean(),
                'CV_Mean_Precision': cv_results['test_precision'].mean(),
                'CV_Mean_Recall': cv_results['test_recall'].mean(),
                'CV_Mean_ROC_AUC': cv_results['test_roc_auc'].mean(),
                'Val_Accuracy': accuracy_score(y_val, y_val_pred_binary),
                'Val_F1': f1_score(y_val, y_val_pred_binary),
                'Val_Precision': precision_score(y_val, y_val_pred_binary),
                'Val_Recall': recall_score(y_val, y_val_pred_binary),
                'Val_ROC_AUC': roc_auc_score(y_val, y_val_pred_binary)
            })
    
    # Run the parallel processing on each DataFrame (with 6 threads)
    Parallel(n_jobs=6)(delayed(process_fingerprint)(idx, df, filename) for idx, (df, filename) in enumerate(zip(fingerprint_dfs, filenames)))

    results_df = pd.DataFrame(results_list)
    return results_df