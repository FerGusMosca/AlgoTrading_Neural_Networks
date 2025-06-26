import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import adfuller
import joblib
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Union
from common.util.light_logger import LightLogger

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
_OUTPUT_PATH = "./output/"

class BaseModelCreator:

    def __add_trading_prices__(self, test_series_df, result_df, symbol, dates, closing_price_col):
        # Asegúrate de que 'symbol' esté presente en test_series_df
        if symbol in test_series_df.columns:
            # Crear un DataFrame temporal con 'date' y la columna del símbolo para los precios de cierre
            temp_df = test_series_df[['date', symbol]].copy()

            # Renombrar la columna del símbolo para que coincida con el nombre deseado de la columna de precios de cierre
            temp_df.rename(columns={symbol: closing_price_col}, inplace=True)

            # Convertir la columna 'date' a formato datetime en ambos DataFrames
            temp_df['date'] = pd.to_datetime(temp_df['date'], unit='s')
            result_df['date'] = pd.to_datetime(result_df['date'], unit='s')

            # Unir result_df con temp_df basándose en la columna 'date' para agregar los precios de cierre
            result_df = result_df.merge(temp_df, on='date', how='left')

            return result_df


        else:
            raise Exception("Missing column {} in df test_series_df ".format(symbol))

    def __get_test_sets_with_scaler__(self, df, variables_csv, model_to_use):
        """
        Load and apply the scaler saved during training to the test data.

        Parameters:
        - df: pd.DataFrame, test data (already preformatted and stationary if needed)
        - variables_csv: str, comma-separated list of feature names
        - model_to_use: str, path to the saved model .pkl (used to locate corresponding scaler)

        Returns:
        - np.ndarray: normalized test data ready for prediction
        """
        # Load the scaler saved along with the model
        scaler_filename = model_to_use.replace(".pkl", "_scaler.pkl")
        scaler = joblib.load(scaler_filename)

        # Extract the expected feature columns
        expected_features = variables_csv.split(',')
        X = df[expected_features].values

        # Validate dimensions
        if X.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: X has {X.shape[1]} columns, but scaler expects {scaler.n_features_in_}")

        # Normalize using the scaler
        X_scaled = scaler.transform(X)
        return X_scaled

    def __get_test_sets__(self, test_series_df, symbol_col='trading_symbol', date_col='date',
                          variables_csv=None,normalize=True):
        """
        Prepare the test set from the given DataFrame.

        Parameters:
        test_series_df (pd.DataFrame): DataFrame containing the time series test data.
        symbol_col (str): The column name for the trading symbol.
        date_col (str): The column name for the date.
        variables_csv (str): Comma-separated string of variable names to use as features.

        Returns:
        np.ndarray: X_test (normalized test features)
        """
        # Print the columns and the first few rows of test_series_df for debugging
        print(f"Columns in test_series_df: {test_series_df.columns.tolist()}")
        print(f"First 5 rows of test_series_df:\n{test_series_df.head()}")

        # Split variables_csv into a list of feature columns
        if variables_csv is None:
            raise ValueError("variables_csv must be provided to determine the feature columns.")
        expected_features = variables_csv.split(',')

        # Exclude symbol_col, date_col, and any columns not in expected_features
        feature_columns = [col for col in test_series_df.columns if col in expected_features]

        # Check for missing or extra features
        missing_features = [col for col in expected_features if col not in feature_columns]
        extra_features = [col for col in test_series_df.columns if
                          col not in expected_features and col not in [symbol_col, date_col]]

        if missing_features:
            raise ValueError(f"Missing columns in test_series_df: {missing_features}")
        if extra_features:
            print(f"Warning: Extra columns in test_series_df that will not be used: {extra_features}")

        # Create X_test with the feature columns
        X_test = test_series_df[feature_columns].values
        print(f"Shape of X_test before normalization: {X_test.shape}")
        print(f"Columns in X_test: {feature_columns}")

        # Normalize the feature data
        if normalize:
            X_test = self.__load_scaler_and_normalize__(X_test)

        return X_test


    def __preformat_test_sets__(self, test_series_df):
        """
        Preformat the test set by dropping NaN values.

        Parameters:
        test_series_df (pd.DataFrame): DataFrame containing the test time series data.
        """
        test_series_df.dropna(inplace=True)

    def __load_rf_model_bundle__(self,model_path: str) -> Tuple[
        RandomForestClassifier, Union[object, None], Union[object, None]]:
        """
        Loads a Random Forest model and its components (label_encoder, scaler) if available.

        Returns:
            - model: RandomForestClassifier
            - label_encoder: Optional, if present in bundle
            - scaler: Optional, if present in bundle
        """
        try:
            loaded_obj = load(model_path)

            # If it's a bundle dict
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                model = loaded_obj["model"]
                label_encoder = loaded_obj.get("label_encoder", None)
                scaler = loaded_obj.get("scaler", None)
                return model, label_encoder, scaler

            # If it's just a RandomForestClassifier
            elif isinstance(loaded_obj, RandomForestClassifier):
                return loaded_obj, None, None

            else:
                raise ValueError(f"Unrecognized format for model file: {model_path}")

        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {e}")

    def __make_stationary_with_memory__(self, df, state=None):
        """
        Make the time series stationary by differencing, using the list of columns
        previously identified as non-stationary (from differenced_columns.csv).
        Maintain continuity across blocks using a state.

        Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        state (dict): Last values from the previous block for continuity (default: None).

        Returns:
        tuple: (stationary DataFrame, updated state)
        """
        # Load the list of columns that need differencing
        output_path = os.path.join(_OUTPUT_PATH, 'differenced_columns.csv')
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Expected differenced_columns.csv at {output_path}. Run __make_stationary__ first.")

        differenced_columns = pd.read_csv(output_path)['differenced_columns'].tolist()
        print(f"Columns to be differenced (loaded from CSV): {differenced_columns}")

        # Apply differencing to the specified columns
        for col in df.columns:
            if col in differenced_columns:
                # If there's a state, use the last value to compute the first difference
                if state is not None and col in state:
                    first_value = df[col].iloc[0]
                    df[col].iloc[0] = first_value - state[col]

                # Apply differencing to the rest of the column
                df[col] = df[col].diff().fillna(0)
                print(f"Applied differencing to {col} with continuity.")
            else:
                print(f"Skipping {col} (not in differenced columns).")

        # Update the state with the last values of the current block
        numeric_cols = df.select_dtypes(include=np.number).columns
        new_state = {col: df[col].iloc[-1] for col in numeric_cols if col in differenced_columns}

        return df, new_state


    # Function to make all numeric variables in the dataframe stationary
    def __make_stationary__(self, df):
        """
        Make the time series stationary by differencing if necessary, based on the ADF test.
        Save the list of columns that were differenced to a CSV file.

        Parameters:
        df (pd.DataFrame): DataFrame with time series data.

        Returns:
        pd.DataFrame: Stationary DataFrame.
        """

        # Inside __make_stationary__ or preprocessing
        for var in df.columns:
            series = df[var].dropna()
            if len(series.unique()) <= 1:
                LightLogger.do_log(f"[WARNING] Variable {var} is constant or has no variation. Skipping...")
                df.drop(columns=[var], inplace=True)

        # List to store columns that were differenced
        differenced_columns = []

        for var in df.columns:
            # Check if the column is numeric (ignore non-numeric columns like dates)
            if pd.api.types.is_numeric_dtype(df[var]):
                # Perform the Augmented Dickey-Fuller (ADF) test to check if the series is stationary
                result = adfuller(df[var])
                print(f"ADF Statistic for {var}: {result[0]}, p-value: {result[1]}")

                # If the p-value is greater than 0.05, the series is non-stationary and needs differencing
                if result[1] > 0.05:
                    # Apply first-order differencing to make the series stationary
                    df[var] = df[var].diff().dropna()
                    differenced_columns.append(var)
                    print(f"Variable {var} has been made stationary using differencing.")
                else:
                    print(f"Variable {var} is already stationary.")
            else:
                print(f"Skipping non-numeric column: {var}")

        # Save the list of differenced columns to a CSV file
        if differenced_columns:
            differenced_df = pd.DataFrame({'differenced_columns': differenced_columns})
            output_path = os.path.join(_OUTPUT_PATH, 'differenced_columns.csv')
            differenced_df.to_csv(output_path, index=False)
            print(f"Differenced columns saved to {output_path}")

        return df

    def __normalize_and_save_scaler__(self, X):
        """
        Normalize the input data using StandardScaler and save the scaler.

        Parameters:
        X (np.ndarray): Input data to normalize.

        Returns:
        np.ndarray: Normalized data.
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Check for NaN or infinite values after normalization
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values after normalization.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values after normalization.")

        # Calculate and print mean and standard deviation for each column
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        print(f"Mean of each column after normalization: {means}")
        print(f"Std of each column after normalization: {stds}")

        # Save the scaler
        joblib.dump(scaler, f"{_OUTPUT_PATH}last_scaler.pkl")
        print(f"Scaler saved to {_OUTPUT_PATH}last_scaler.pkl")

        return X

    def __normalize_and_use_scaler(self,scaler,X):
        X = scaler.transform(X)
        return X

    def __get_training_sets__(self, df, symbol_col, date_col, classif_key, variables_csv, test_size,
                              return_encoder_and_scaler=False):
        expected_features = variables_csv.split(',')

        train_size = int(len(df) * (1 - test_size))
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Feature matrices
        X_train_df = train_df[expected_features]
        X_test_df = test_df[expected_features]

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test = scaler.transform(X_test_df)

        # Encode labels
        y_train_raw = train_df[classif_key].values
        y_test_raw = test_df[classif_key].values

        label_encoder = LabelEncoder()
        present_classes = np.unique(df[classif_key])
        label_encoder.fit(present_classes.tolist())
        # Fit with all classes to avoid unseen error
        y_train = label_encoder.transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)

        # Return tuple based on flag
        if return_encoder_and_scaler:
            return X_train, X_test, y_train, y_test, label_encoder, scaler
        else:
            return X_train, X_test, y_train, y_test

    def __preformat_training_set__(self,training_series_df):

        training_series_df.dropna(inplace=True)



    def __load_scaler_and_normalize__(self, X):
        """
        Load the saved scaler and normalize the input data.

        Parameters:
        X (np.ndarray): Input data to normalize.

        Returns:
        np.ndarray: Normalized data.
        """
        # Load the scaler
        scaler = joblib.load(f"{_OUTPUT_PATH}last_scaler.pkl")
        print(f"Shape of X before transformation: {X.shape}")
        print(f"Expected columns by the scaler: {scaler.n_features_in_}")

        # Check for dimension mismatch
        if X.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: X has {X.shape[1]} columns, but the scaler expects {scaler.n_features_in_} columns.")

        # Check for NaN or non-numeric values before transformation
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values before transformation.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values before transformation.")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X contains non-numeric values before transformation.")

        # Transform the data
        X = scaler.transform(X)

        # Check for NaN or infinite values after normalization
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values after normalization.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values after normalization.")

        # Calculate and print mean and standard deviation for each column
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        print(f"Mean of each column after normalization: {means}")
        print(f"Std of each column after normalization: {stds}")

        return X