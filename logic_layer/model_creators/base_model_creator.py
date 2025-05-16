import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import adfuller

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
_OUTPUT_PATH = "./output/"

class BaseModelCreator:

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

    def __get_training_sets__(self, df, symbol_col, date_col, classif_key, variables_csv, test_size):
        expected_features = variables_csv.split(
            ',')  # Lista de variables esperadas (por ejemplo, ['SPY', 'XLY', 'VIXCLS', ...])
        train_size = int(len(df) * (1 - test_size))  # Calcula el tamaño del conjunto de entrenamiento
        train_df = df[:train_size]  # Divide el DataFrame en entrenamiento (primeros train_size registros)
        test_df = df[train_size:]  # Divide el DataFrame en prueba (registros restantes)
        # Define las columnas a eliminar: columnas de símbolo, fecha, clase, y cualquier columna no esperada
        columns_to_drop = [symbol_col, date_col, classif_key] + [col for col in train_df.columns if
                                                                 col not in expected_features and col not in [
                                                                     symbol_col, date_col, classif_key]]
        X_train = train_df.drop(columns=columns_to_drop).values  # Extrae las características de entrenamiento
        y_train = train_df[classif_key].values  # Extrae las etiquetas de entrenamiento
        X_test = test_df.drop(columns=columns_to_drop).values  # Extrae las características de prueba
        y_test = test_df[classif_key].values  # Extrae las etiquetas de prueba
        print(f"Shape of X_train before normalization: {X_train.shape}")
        print(f"Columns in X_train: {train_df.drop(columns=columns_to_drop).columns.tolist()}")

        print(f"X_test shape: {X_test.shape}")
        print(f"First 5 rows of training_series_df:\n{df.head()}")
        print(f"Last 5 rows of training_series_df:\n{df.tail()}")
        print(f"Split sizes - X_train: {len(X_train)}, X_test: {len(X_test)}")

        X_train = self.__normalize_and_save_scaler__(X_train)  # Normaliza X_train y guarda el escalador
        X_test = self.__load_scaler_and_normalize__(X_test)  # Normaliza X_test usando el escalador guardado
        label_encoder_target = LabelEncoder()  # Codifica las etiquetas (por ejemplo, 'LONG'/'SHORT' a 0/1)
        y_train = label_encoder_target.fit_transform(y_train)
        y_test = label_encoder_target.transform(y_test)
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