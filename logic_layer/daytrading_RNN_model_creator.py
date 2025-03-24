import math
import os

import joblib
import pandas as pd
import numpy as np
import tensorflow
from keras import Sequential
from keras.src.callbacks import Callback, ReduceLROnPlateau
from keras.src.layers import TimeDistributed, BatchNormalization, InputLayer
from keras.src.optimizers import Adam
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.utils import compute_class_weight
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from common.util.dataframe_filler import DataframeFiller
from common.util.machine_learning.f1_score import F1Score
from imblearn.under_sampling import RandomUnderSampler

_OUTPUT_DATE_FORMAT='%m/%d/%Y %H:%M:%S'
_OUTPUT_PATH="./output/"

class CustomEarlyStopping(Callback):
    def __init__(self, monitor_train='loss', monitor_val='val_loss', threshold=0.8, patience=5):
        """
        Custom Early Stopping callback to stop training when both training and validation metrics
        are below (for loss) or above (for accuracy) a threshold for a number of epochs (patience).

        Parameters:
        monitor_train (str): Metric to monitor for training (e.g., 'loss', 'accuracy').
        monitor_val (str): Metric to monitor for validation (e.g., 'val_loss', 'val_accuracy').
        threshold (float): Threshold value to stop training.
        patience (int): Number of epochs to wait after threshold is reached.
        """
        super(CustomEarlyStopping, self).__init__()
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.threshold = threshold
        self.patience = patience
        self.epochs_waited = 0
        self.best_train = float('inf') if 'loss' in monitor_train else -float('inf')
        self.best_val = float('inf') if 'loss' in monitor_val else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        train_metric = logs.get(self.monitor_train)
        val_metric = logs.get(self.monitor_val)

        print(f"Evaluating early stop for {self.monitor_train}={train_metric} {self.monitor_val}={val_metric}")

        # Determine if the metric should be minimized (loss) or maximized (accuracy)
        is_loss = 'loss' in self.monitor_train.lower()
        train_condition = train_metric <= self.threshold if is_loss else train_metric >= self.threshold
        val_condition = val_metric <= self.threshold if is_loss else val_metric >= self.threshold

        # Check if both conditions are met
        if train_metric is not None and val_metric is not None:
            if train_condition and val_condition:
                self.epochs_waited += 1
                print(f"Threshold reached: {self.monitor_train} = {train_metric:.4f}, {self.monitor_val} = {val_metric:.4f}, waited {self.epochs_waited}/{self.patience} epochs.")
                if self.epochs_waited >= self.patience:
                    print(f"Stopping training after {self.patience} epochs of meeting the threshold.")
                    self.model.stop_training = True
            else:
                self.epochs_waited = 0
class DayTradingRNNModelCreator:

    def __init__(self):
        pass


    #region Private Methods

    def __preformat_test_sets__(self, test_series_df):
        """
        Preformat the test set by dropping NaN values.

        Parameters:
        test_series_df (pd.DataFrame): DataFrame containing the test time series data.
        """
        test_series_df.dropna(inplace=True)

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

    def __preformat_training_set__(self,training_series_df):

        training_series_df.dropna(inplace=True)

    def __get_test_sets_daily__(self, test_series_df, symbol_col='trading_symbol', date_col='date'):
        """
        Prepare the test set from the given DataFrame.

        Parameters:
        test_series_df (pd.DataFrame): DataFrame containing the time series test data.
        symbol_col (str): The column name for the trading symbol.
        date_col (str): The column name for the date.

        Returns:
        np.ndarray: X_test (normalized test features)
        """
        # Preprocess 'trading_symbol' column (convert to numeric using LabelEncoder)
        if symbol_col in test_series_df.columns:
            label_encoder_symbol = LabelEncoder()
            test_series_df[symbol_col] = label_encoder_symbol.fit_transform(test_series_df[symbol_col])

        # Preprocess 'date' column (convert to timestamp)
        if date_col in test_series_df.columns:
            test_series_df[date_col] = pd.to_datetime(test_series_df[date_col])  # Ensure it's a datetime object
            test_series_df[date_col] = test_series_df[date_col].map(pd.Timestamp.timestamp)  # Convert to timestamp

        # Calculate feature columns based on training data (excluding the classification column)
        feature_columns = [col for col in test_series_df.columns]

        # Use the feature columns from training to ensure consistency
        if feature_columns is not None:
            X_test = test_series_df[feature_columns].values
        else:
            raise ValueError("feature_columns must be provided to match the training set.")

        # Normalize the feature data
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        return X_test

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

            return  result_df


        else:
            raise Exception("Missing column {} in df test_series_df ".format(symbol))

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
        X_train = self.__normalize_and_save_scaler__(X_train)  # Normaliza X_train y guarda el escalador
        X_test = self.__load_scaler_and_normalize__(X_test)  # Normaliza X_test usando el escalador guardado
        label_encoder_target = LabelEncoder()  # Codifica las etiquetas (por ejemplo, 'LONG'/'SHORT' a 0/1)
        y_train = label_encoder_target.fit_transform(y_train)
        y_test = label_encoder_target.transform(y_test)
        return X_train, X_test, y_train, y_test

    #endregion


    #region Public Methods

    def train_LSTM(self, training_series_df, model_output, symbol, classif_key, epochs,
                   timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,
                   variables_csv, clipping_rate=None,
                   threshold_stop=None, make_stationary=False, inner_activation='tanh', batch_size=1):
        """
        Build and train an LSTM model on the given training data and save the model using TimeSeriesSplit.

        Parameters:
        training_series_df (pd.DataFrame): DataFrame containing the training data.
        model_output (str): Path where the trained model will be saved.
        classif_key (str): The column name of the classification target.
        epochs (int): Number of epochs to train the model.
        timestamps (int): Number of time steps to look back in the time series.
        n_neurons (int): Number of neurons in each LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
        reg_rate (float): L2 regularization rate.
        dropout_rate (float): Dropout rate for regularization.
        variables_csv (str): Comma-separated list of variables to use.
        clipping_rate (float): Gradient clipping value.
        threshold_stop (float): Threshold for early stopping.
        make_stationary (bool): Whether to make the data stationary.
        inner_activation (str): Activation function for LSTM layers.
        batch_size (int): Batch size for training.
        """
        try:
            # Fill missing values in the DataFrame
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)

            # 1. Make the data stationary if specified
            if make_stationary:
                training_series_df = self.__make_stationary__(training_series_df)

            # 2. Preformat the DataFrame
            self.__preformat_training_set__(training_series_df)

            # 3. Get training and test sets (initial split to get X and y)
            X_train_full, X_test_full, y_train_full, y_test_full = self.__get_training_sets__(training_series_df,
                                                                                              "trading_symbol", "date",
                                                                                              classif_key,
                                                                                              variables_csv=variables_csv,
                                                                                              test_size=0.2)

            # Debug: Print the distribution of classes in the full training set
            print("Distribución de clases en y_train_full (antes de TimeSeriesSplit):", np.bincount(y_train_full))
            print("Distribución de clases en y_test_full:", np.bincount(y_test_full))

            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=5)  # 5 folds for cross-validation
            best_val_f1_global = 0  # Track the best F1-Score across all folds
            best_model = None

            # Lists to store metrics across folds
            val_accuracies = []
            val_f1_scores = []
            val_losses = []  # To monitor overfitting

            for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full)):
                print(f"\nEntrenando fold {fold + 1}/{tscv.n_splits}...")

                # Split the data for this fold
                X_train, X_val = X_train_full[train_index], X_train_full[val_index]
                y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                # Normalize the data for this fold
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
                X_val = scaler.transform(X_val)  # Transform validation data using the same scaler
                print(f"Fold {fold + 1} - Mean of X_train after normalization:", np.mean(X_train, axis=0))
                print(f"Fold {fold + 1} - Std of X_train after normalization:", np.std(X_train, axis=0))
                print(f"Fold {fold + 1} - Mean of X_val after normalization:", np.mean(X_val, axis=0))
                print(f"Fold {fold + 1} - Std of X_val after normalization:", np.std(X_val, axis=0))

                # Check the number of minority class samples
                minority_class_count = np.sum(y_train == 1)
                print(
                    f"Número de muestras de la clase minoritaria en y_train (fold {fold + 1}): {minority_class_count}")

                # Apply RandomUnderSampler to balance the classes (50-50)
                if minority_class_count > 0:  # Only apply undersampling if there are minority class samples
                    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                else:
                    print(
                        f"Fold {fold + 1} - No hay muestras de la clase minoritaria en este fold. Usando datos sin balancear.")
                    X_train_resampled, y_train_resampled = X_train, y_train

                # Print the distribution of classes in this fold
                print(f"Fold {fold + 1} - Distribución de clases en y_train después de submuestreo:",
                      np.bincount(y_train_resampled))
                print(f"Fold {fold + 1} - Distribución de clases en y_val:", np.bincount(y_val))

                # Clip the data to prevent extreme values
                X_train_resampled = np.clip(X_train_resampled, -3, 3)
                X_val = np.clip(X_val, -3, 3)

                # TimeseriesGenerator for training and validation data
                train_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_train_resampled,
                                                                                              y_train_resampled,
                                                                                              length=timestamps,
                                                                                              batch_size=batch_size)
                val_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_val, y_val,
                                                                                            length=timestamps,
                                                                                            batch_size=batch_size)

                # Define class weights
                class_weight_dict = {0: 1.0, 1: 7.0}  # Increased weight for minority class
                print(f"Pesos de clase para fold {fold + 1}: {class_weight_dict}")

                # Define the LSTM model with regularization
                model = Sequential([
                    # First LSTM layer
                    LSTM(n_neurons,
                         activation=inner_activation,
                         return_sequences=True,
                         input_shape=(timestamps, X_train.shape[1]),
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),

                    # Second LSTM layer
                    LSTM(n_neurons,
                         activation=inner_activation,
                         return_sequences=True,
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),

                    # Third LSTM layer
                    LSTM(n_neurons,
                         activation=inner_activation,
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),

                    # Output layer
                    Dense(1,
                          activation='sigmoid',
                          kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate))
                ])

                # Define focal loss to handle class imbalance
                def focal_loss(gamma=3.0, alpha=0.5):  # Increased gamma to 3.0
                    def focal_loss_fixed(y_true, y_pred):
                        y_true = tensorflow.cast(y_true, tensorflow.float32)
                        y_pred = tensorflow.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
                        return -tensorflow.reduce_mean(
                            alpha_t * tensorflow.pow(1.0 - pt, gamma) * tensorflow.math.log(pt))

                    return focal_loss_fixed

                # Adjust the learning rate and apply gradient clipping
                optimizer = Adam(learning_rate=learning_rate,
                                 clipvalue=clipping_rate if clipping_rate is not None else 15.0)

                # Compile the model with focal loss and F1-Score metric
                model.compile(optimizer=optimizer,
                              loss=focal_loss(gamma=3.0, alpha=0.5),
                              metrics=['accuracy', F1Score()])

                # Early stopping based on F1-Score
                early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                                          patience=20,  # Increased patience
                                                                          mode='max',
                                                                          restore_best_weights=True)

                # Learning rate scheduler
                lr_scheduler = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=1e-6)

                # Train the model for this fold
                history = model.fit(train_generator,
                                    epochs=epochs,
                                    validation_data=val_generator,
                                    verbose=1,
                                    callbacks=[early_stopping, lr_scheduler],
                                    class_weight=class_weight_dict)

                # Get the raw probability predictions for this fold (on validation set)
                val_probabilities = model.predict(val_generator).flatten()

                # Use a fixed threshold to favor the minority class
                fixed_threshold = 0.4  # Lowered threshold to favor class 1
                val_predictions = (val_probabilities > fixed_threshold).astype(int)
                print(f"Umbral fijo para fold {fold + 1}: {fixed_threshold}")
                print(f"Distribución de predicciones en y_val (fold {fold + 1}):", np.bincount(val_predictions))

                # Get the best validation metrics for this fold
                best_val_accuracy = max(history.history['val_accuracy'])
                best_val_f1 = max(history.history['val_f1_score'])
                best_val_loss = min(history.history['val_loss'])
                val_accuracies.append(best_val_accuracy)
                val_f1_scores.append(best_val_f1)
                val_losses.append(best_val_loss)
                print(f"Mejor val_accuracy en fold {fold + 1}: {best_val_accuracy}")
                print(f"Mejor val_f1_score en fold {fold + 1}: {best_val_f1}")
                print(f"Mejor val_loss en fold {fold + 1}: {best_val_loss}")

                # Save the model if it has the best F1-Score so far
                if best_val_f1 > best_val_f1_global:
                    best_val_f1_global = best_val_f1
                    best_model = model

            # Print average metrics across folds
            print(f"Promedio de val_accuracy: {np.mean(val_accuracies)}")
            print(f"Promedio de val_f1_score: {np.mean(val_f1_scores)}")
            print(f"Promedio de val_loss: {np.mean(val_losses)}")

            # Check for overfitting by comparing the trend of val_loss
            if len(val_losses) > 1 and val_losses[-1] > val_losses[-2]:
                print("Advertencia: val_loss está aumentando, posible sobreajuste detectado.")

            # Save the best model
            if best_model is not None:
                best_model.save(model_output)
                print(f"Modelo guardado en {model_output}")
            else:
                raise Exception("No se entrenó ningún modelo exitosamente.")

        except Exception as e:
            raise Exception(f"Error al construir el modelo LSTM para el símbolo {symbol}: {e}")


    def preload_model(self,model_to_use):
        # Load the saved LSTM model
        model = tensorflow.keras.models.load_model(model_to_use)
        return  model

    def test_stateful_LSTM(self, symbol, test_series_df, model_to_use, timesteps, price_to_use="close",
                                      preloaded_model=None):
        self.__preformat_test_sets__(test_series_df)

        # Prepare the test dataset
        X_test = self.__get_test_sets__(test_series_df, symbol_col="trading_symbol", date_col="date")

        # Load the LSTM model
        if preloaded_model is None:
            model = tensorflow.keras.models.load_model(model_to_use)
        else:
            model = preloaded_model

        # Create a time series generator for sequential test data
        test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(
            X_test, np.zeros(len(X_test)), length=timesteps, batch_size=1
        )

        # Perform predictions iteratively
        predictions = []
        for i in range(len(test_generator)):
            X_batch, _ = test_generator[i]  # Get the current batch (1 timestep window)
            pred = model.predict(X_batch, batch_size=1)  # No states involved
            predictions.append(pred)

        # Convert predictions to actions (e.g., LONG, SHORT, FLAT)
        predictions = np.vstack(predictions)  # Combine predictions into a single array
        actions = np.argmax(predictions, axis=1)

        # Map actions to readable labels
        action_labels = {0: "LONG", 1: "SHORT", 2: "FLAT"}
        action_series = pd.Series(actions).map(action_labels)

        # Adjust the DataFrame to match prediction length
        dates = pd.to_datetime(test_series_df['date'].iloc[timesteps:].reset_index(drop=True), unit='s')
        formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)

        # Create the final result DataFrame
        result_df = pd.DataFrame({
            'trading_symbol': symbol,
            'date': dates,
            'formatted_date': formatted_dates,
            'action': action_series
        })

        # Add trading prices for better analysis
        result_df = self.__add_trading_prices__(test_series_df, result_df, f"{price_to_use}_{symbol}", dates,
                                                "trading_symbol_price")

        return result_df


    def test_LSTM_daily(self,symbol,test_series_df, model_to_use,timesteps,price_to_use="close",
                            preloaded_model=None, prev_states=None,variables_csv=None):

        self.__preformat_test_sets__(test_series_df)

        X_test=self.__get_test_sets_daily__(test_series_df,symbol_col="trading_symbol",date_col="date")

        model=None
        if(preloaded_model is None):
            # Load the saved LSTM model
            model = tensorflow.keras.models.load_model(model_to_use)
        else:
            model=preloaded_model

        # timestamps= Number of timestamps used in training (adjust this to match your model's training configuration)
        # Create a time series generator for the test data
        test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(
            X_test, np.zeros(len(X_test)), length=timesteps, batch_size=1
        )

        # Generate predictions
        states=None
        if prev_states is None:
            predictions = model.predict(test_generator)
        else:
            predictions = model.predict(test_generator,prev_states)

        # Convert predictions to actions (LONG, SHORT, FLAT)
        actions = np.argmax(predictions, axis=1)

        action_labels = {0: "LONG", 1: "SHORT", 2: "FLAT"}
        action_series = pd.Series(actions).map(action_labels)

        # Adjust the DataFrame to match the length of the predictions
        dates = pd.to_datetime(test_series_df['date'].iloc[timesteps:].reset_index(drop=True), unit='s')
        formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)
        #symbols = test_series_df['trading_symbol'].iloc[timestamps:].reset_index(drop=True)

        # Create the final output DataFrame
        result_df = pd.DataFrame({
            'trading_symbol': symbol,
            'date': dates,
            'formatted_date': formatted_dates,
            'action': action_series
        })


        result_df=self.__add_trading_prices__(test_series_df,result_df,f"{price_to_use}_{symbol}",dates,"trading_symbol_price")


        return result_df,states

    def test_LSTM_scalping(self, symbol, test_series_df, model_to_use, timesteps, price_to_use="close",
                  preloaded_model=None, prev_states=None, make_stationary=True,
                  normalize=True, variables_csv=None,threshold=0.6):
        """
        Test the LSTM model on the given data, with a custom threshold for predictions.

        Parameters:
        threshold (float): Probability threshold for predicting LONG (default: 0.6).
                           If the probability of LONG is below this threshold, predict SHORT.
        """
        if make_stationary:
            test_series_df, stationarity_state = self.__make_stationary_with_memory__(test_series_df,
                                                                                      state=prev_states.get(
                                                                                          'stationarity_state',
                                                                                          None) if prev_states else None)
        else:
            stationarity_state = None

        self.__preformat_test_sets__(test_series_df)

        X_test = self.__get_test_sets__(test_series_df, symbol_col="trading_symbol", date_col="date",
                                        variables_csv=variables_csv,normalize=normalize)

        model = None
        if preloaded_model is None:
            model = tensorflow.keras.models.load_model(model_to_use)
        else:
            model = preloaded_model

        test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(
            X_test, np.zeros(len(X_test)), length=timesteps, batch_size=1
        )

        states = {}
        if prev_states is None or 'model_states' not in prev_states:
            predictions = model.predict(test_generator)
        else:
            predictions = model.predict(test_generator, prev_states['model_states'])

        # Apply a custom threshold to predictions instead of np.argmax
        # predictions[:, 0] is the probability of LONG, predictions[:, 1] is the probability of SHORT
        actions = np.where(predictions[:, 0] >= threshold, 0, 1)  # 0 for LONG, 1 for SHORT
        action_labels = {0: "LONG", 1: "SHORT"}
        action_series = pd.Series(actions).map(action_labels)

        dates = pd.to_datetime(test_series_df['date'].iloc[timesteps:].reset_index(drop=True), unit='s')
        formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)

        result_df = pd.DataFrame({
            'trading_symbol': symbol,
            'date': dates,
            'formatted_date': formatted_dates,
            'action': action_series
        })

        result_df = self.__add_trading_prices__(test_series_df, result_df, f"{price_to_use}_{symbol}", dates,
                                                "trading_symbol_price")

        states['model_states'] = model.get_states() if hasattr(model, 'get_states') else None
        states['stationarity_state'] = stationarity_state

        return result_df, states

    #endregion