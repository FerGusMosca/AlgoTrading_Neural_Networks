import pandas as pd
import numpy as np
import tensorflow
from keras import Sequential, Input, Model
from keras.src.callbacks import Callback, ReduceLROnPlateau
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.machine_learning.f1_score import F1Score
from imblearn.under_sampling import RandomUnderSampler

from logic_layer.model_creators.base_model_creator import BaseModelCreator

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
class DayTradingRNNModelCreator(BaseModelCreator):

    def __init__(self):
        pass


    #region Private Methods

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

    def __get_test_sets_from_model__(self, test_series_df, model,
                                     symbol_col='trading_symbol', date_col='date'):
        """
        Prepare test set exactly like __get_test_sets_daily__, using the saved scaler.

        Parameters:
            test_series_df (pd.DataFrame): DataFrame with test data
            model (keras.Model): Already loaded model (used only to infer shape)
            symbol_col (str): Column name for symbol
            date_col (str): Column name for date

        Returns:
            np.ndarray: X_test normalized and ready for inference
        """

        # Preprocess trading symbol (LabelEncoder)
        if symbol_col in test_series_df.columns:
            from sklearn.preprocessing import LabelEncoder
            label_encoder_symbol = LabelEncoder()
            test_series_df[symbol_col] = label_encoder_symbol.fit_transform(test_series_df[symbol_col])

        # Preprocess date to timestamp
        if date_col in test_series_df.columns:
            test_series_df[date_col] = pd.to_datetime(test_series_df[date_col])
            test_series_df[date_col] = test_series_df[date_col].map(pd.Timestamp.timestamp)

        # Use all columns in test_series_df as features
        feature_columns = [col for col in test_series_df.columns]
        X_test = test_series_df[feature_columns].values

            # Normalize the feature data
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        return X_test


    #endregion


    #region Public Methods

    def train_LSTM_scalping(self, training_series_df, model_output, symbol, classif_key, epochs,
                            timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,
                            variables_csv, clipping_rate=None,
                            threshold_stop=None, make_stationary=False, inner_activation='tanh', batch_size=1):
        try:
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)

            if make_stationary:
                training_series_df = self.__make_stationary__(training_series_df)

            self.__preformat_training_set__(training_series_df)

            X_train_full, X_test_full, y_train_full, y_test_full = self.__get_training_sets__(training_series_df,
                                                                                              "trading_symbol", "date",
                                                                                              classif_key,
                                                                                              variables_csv=variables_csv,
                                                                                              test_size=0.2)

            print("Distribución de clases en y_train_full (antes de TimeSeriesSplit):", np.bincount(y_train_full))
            print("Distribución de clases en y_test_full:", np.bincount(y_test_full))

            tscv = TimeSeriesSplit(n_splits=3)
            best_val_f1_global = 0
            best_model = None

            val_accuracies = []
            val_f1_scores = []
            val_losses = []

            for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full)):
                print(f"\nEntrenando fold {fold + 1}/{tscv.n_splits}...")

                X_train, X_val = X_train_full[train_index], X_train_full[val_index]
                y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                print(f"Fold {fold + 1} - Mean of X_train after normalization:", np.mean(X_train, axis=0))
                print(f"Fold {fold + 1} - Std of X_train after normalization:", np.std(X_train, axis=0))
                print(f"Fold {fold + 1} - Mean of X_val after normalization:", np.mean(X_val, axis=0))
                print(f"Fold {fold + 1} - Std of X_val after normalization:", np.std(X_val, axis=0))

                minority_class_count = np.sum(y_train == 1)
                print(
                    f"Número de muestras de la clase minoritaria en y_train (fold {fold + 1}): {minority_class_count}")

                if minority_class_count > 0:
                    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                else:
                    print(
                        f"Fold {fold + 1} - No hay muestras de la clase minoritaria en este fold. Usando datos sin balancear.")
                    X_train_resampled, y_train_resampled = X_train, y_train

                print(f"Fold {fold + 1} - Distribución de clases en y_train después de submuestreo:",
                      np.bincount(y_train_resampled))
                print(f"Fold {fold + 1} - Distribución de clases en y_val:", np.bincount(y_val))

                X_train_resampled = np.clip(X_train_resampled, -3, 3)
                X_val = np.clip(X_val, -3, 3)

                train_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_train_resampled,
                                                                                              y_train_resampled,
                                                                                              length=timestamps,
                                                                                              batch_size=batch_size)
                val_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_val, y_val,
                                                                                            length=timestamps,
                                                                                            batch_size=batch_size)

                class_weight_dict = {0: 1.0, 1: 2.0}  # Manteniendo en 2.0 por ahora
                print(f"Pesos de clase para fold {fold + 1}: {class_weight_dict}")

                model = Sequential([
                    LSTM(n_neurons,  # Aumentado a 16 en el comando
                         activation=inner_activation,
                         return_sequences=True,
                         input_shape=(timestamps, X_train.shape[1]),
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),  # Reducido a 0.7 en el comando

                    LSTM(n_neurons,
                         activation=inner_activation,
                         return_sequences=True,
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),

                    LSTM(n_neurons,
                         activation=inner_activation,
                         kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate)),
                    BatchNormalization(),
                    Dropout(dropout_rate),

                    Dense(1,
                          activation='sigmoid',
                          kernel_regularizer=tensorflow.keras.regularizers.l2(reg_rate))
                ])

                def focal_loss(gamma=3.0, alpha=0.75):
                    def focal_loss_fixed(y_true, y_pred):
                        y_true = tensorflow.cast(y_true, tensorflow.float32)
                        y_pred = tensorflow.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
                        return -tensorflow.reduce_mean(
                            alpha_t * tensorflow.pow(1.0 - pt, gamma) * tensorflow.math.log(pt))

                    return focal_loss_fixed

                optimizer = Adam(learning_rate=learning_rate,  # Aumentado a 2e-5 en el comando
                                 clipvalue=clipping_rate if clipping_rate is not None else 15.0)

                model.compile(optimizer=optimizer,
                              loss=focal_loss(gamma=3.0, alpha=0.75),
                              metrics=['accuracy', F1Score()])

                early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                                          patience=20,
                                                                          mode='max',
                                                                          restore_best_weights=True)

                lr_scheduler = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=1e-7)

                history = model.fit(train_generator,
                                    epochs=epochs,
                                    validation_data=val_generator,
                                    verbose=1,
                                    callbacks=[early_stopping, lr_scheduler],
                                    class_weight=class_weight_dict)

                val_probabilities = model.predict(val_generator).flatten()

                # Ajustar dinámicamente el umbral para maximizar el F1-score, priorizando precisión
                from sklearn.metrics import f1_score
                thresholds = np.arange(0.6, 0.9, 0.05)  # Aumentado el rango mínimo a 0.6
                best_f1 = 0
                best_threshold = 0.6
                for threshold in thresholds:
                    val_predictions = (val_probabilities > threshold).astype(int)
                    f1 = f1_score(y_val[timestamps:], val_predictions)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

                val_predictions = (val_probabilities > best_threshold).astype(int)
                print(f"Mejor umbral para fold {fold + 1}: {best_threshold}")
                print(f"Distribución de predicciones en y_val (fold {fold + 1}):", np.bincount(val_predictions))

                # Calcular y mostrar la matriz de confusión
                from sklearn.metrics import confusion_matrix, precision_score, recall_score
                cm = confusion_matrix(y_val[timestamps:], val_predictions)
                print(f"Matriz de confusión para fold {fold + 1}:")
                print(cm)

                # Calcular precisión y recall para la clase 1
                precision = precision_score(y_val[timestamps:], val_predictions, pos_label=1, zero_division=0)
                recall = recall_score(y_val[timestamps:], val_predictions, pos_label=1, zero_division=0)
                print(f"Precisión para la clase 1 en fold {fold + 1}: {precision:.4f}")
                print(f"Recall para la clase 1 en fold {fold + 1}: {recall:.4f}")

                # Mejorar el cálculo del retorno acumulado
                val_dates = training_series_df.index[val_index]
                val_prices = training_series_df.loc[val_dates, 'close_XLK'].values
                val_prices = val_prices[timestamps:]  # Recortar los primeros `timestamps`

                # Inspeccionar val_prices antes de calcular los retornos
                print(f"Fold {fold + 1} - val_prices (primeros 10 valores):", val_prices[:10])
                print(f"Fold {fold + 1} - Número de ceros en val_prices:", np.sum(val_prices == 0))
                print(f"Fold {fold + 1} - Número de NaN en val_prices:", np.sum(np.isnan(val_prices)))

                # Asegurarse de que val_prices y val_predictions tengan la misma longitud
                if len(val_prices) != len(val_predictions):
                    raise ValueError(
                        f"Longitudes no coinciden: val_prices tiene {len(val_prices)} elementos, val_predictions tiene {len(val_predictions)} elementos.")

                # Filtrar días con val_prices igual a 0
                valid_indices = val_prices != 0
                val_prices_filtered = val_prices[valid_indices]
                val_predictions_filtered = val_predictions[valid_indices]

                # Rellenar NaN
                val_prices_filtered = pd.Series(val_prices_filtered).ffill().bfill().values

                # Verificar que no haya NaN después del llenado
                if np.any(np.isnan(val_prices_filtered)):
                    print(f"Advertencia: val_prices_filtered contiene NaN después del llenado en fold {fold + 1}")
                    val_prices_filtered = np.where(np.isnan(val_prices_filtered), 1.0, val_prices_filtered)

                # Calcular retornos evitando divisiones por cero
                returns = np.zeros(len(val_prices_filtered) - 1)
                for i in range(len(returns)):
                    if val_prices_filtered[i] != 0 and not np.isnan(val_prices_filtered[i]) and not np.isnan(
                            val_prices_filtered[i + 1]):
                        returns[i] = (val_prices_filtered[i + 1] - val_prices_filtered[i]) / val_prices_filtered[i]
                        # Clip retornos para evitar valores extremos
                        returns[i] = np.clip(returns[i], -1, 1)
                    else:
                        returns[i] = 0

                # Verificar que no haya NaN en los retornos
                if np.any(np.isnan(returns)):
                    print(f"Advertencia: returns contiene NaN en fold {fold + 1}")
                    returns = np.nan_to_num(returns, 0.0)

                # Ajustar la estrategia de trading: mantener la posición por 1 día
                strategy_returns = np.zeros(len(returns))
                hold_period = 1  # Manteniendo en 1 día
                i = 0
                while i < len(returns):
                    if val_predictions_filtered[i] == 1:
                        # Calcular el retorno acumulado durante el período de tenencia
                        cumulative_return = 0
                        for j in range(min(hold_period, len(returns) - i)):
                            cumulative_return += returns[i + j]
                        # Asignar el retorno acumulado a la primera posición
                        strategy_returns[i] = cumulative_return
                        i += hold_period  # Saltar al final del período de tenencia
                    else:
                        strategy_returns[i] = 0
                        i += 1

                cumulative_return = np.prod(1 + strategy_returns) - 1
                print(f"Retorno acumulado para fold {fold + 1}: {cumulative_return * 100:.2f}%")

                best_val_accuracy = max(history.history['val_accuracy'])
                best_val_f1 = max(history.history['val_f1_score'])
                best_val_loss = min(history.history['val_loss'])
                val_accuracies.append(best_val_accuracy)
                val_f1_scores.append(best_val_f1)
                val_losses.append(best_val_loss)
                print(f"Mejor val_accuracy en fold {fold + 1}: {best_val_accuracy}")
                print(f"Mejor val_f1_score en fold {fold + 1}: {best_val_f1}")
                print(f"Mejor val_loss en fold {fold + 1}: {best_val_loss}")

                if best_val_f1 > best_val_f1_global:
                    best_val_f1_global = best_val_f1
                    best_model = model

            print(f"Promedio de val_accuracy: {np.mean(val_accuracies)}")
            print(f"Promedio de val_f1_score: {np.mean(val_f1_scores)}")
            print(f"Promedio de val_loss: {np.mean(val_losses)}")

            if len(val_losses) > 1 and val_losses[-1] > val_losses[-2]:
                print("Advertencia: val_loss está aumentando, posible sobreajuste detectado.")

            if best_model is not None:
                best_model.save(model_output)
                print(f"Modelo guardado en {model_output}")
            else:
                raise Exception("No se entrenó ningún modelo exitosamente.")

        except Exception as e:
            raise Exception(f"Error al construir el modelo LSTM para el símbolo {symbol}: {e}")


    '''
    def train_LSTM_daily_unstateful(self, training_series_df, model_output, symbol, classif_key, epochs,
                              timestamps, n_neurons, learning_rate, reg_rate, dropout_rate, clipping_rate=None,
                              accuracy_stop=None, inner_activation='tanh', batch_size=1):
        """
        Build and train an LSTM model on the given training data and save the model.

        Parameters:
        training_series_df (pd.DataFrame): DataFrame containing the training data.
        model_output (str): Path where the trained model will be saved.
        classif_key (str): The column name of the classification target.
        safety_minutes (int): Number of time steps to look back in the time series.
        """
        try:

            training_series_df = DataframeFiller.fill_missing_values(training_series_df)

            self.__preformat_training_set__(training_series_df)
            # Get training and test sets
            X_train, X_test, y_train, y_test = self.__get_training_sets__(training_series_df,
                                                                          "trading_symbol", "date",
                                                                          classif_key,variables_csv="",test_size=-0.2
                                                                          )

            print("X_Train: NaN={} Inf={}".format(np.isnan(X_train).sum(), np.isinf(X_train).sum()))

            # number of timestamps to use
            timesteps = timestamps

            # Generador de series temporales para datos de entrenamiento y prueba
            train_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_train, y_train,
                                                                                          length=timesteps,
                                                                                          batch_size=batch_size)
            test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_test, y_test,
                                                                                         length=timesteps,
                                                                                         batch_size=batch_size)

            # Define the LSTM model
            model = tensorflow.keras.models.Sequential()
            model.add(
                LSTM(n_neurons, activation=inner_activation, return_sequences=True,
                     input_shape=(timesteps, X_train.shape[1])))
            model.add(Dropout(dropout_rate))  # Dropout layer with 20% dropout rate
            model.add(BatchNormalization())
            model.add(LSTM(n_neurons, activation=inner_activation))  # Another LSTM layer without return_sequences
            model.add(Dropout(dropout_rate))  # Dropout layer with 20% dropout rate
            model.add(BatchNormalization())
            model.add(Dense(3, activation='softmax',
                            kernel_regularizer=tensorflow.keras.regularizers.l2(
                                reg_rate)))  # Three classes: LONG, SHORT, FLAT

            # Adjust the learning rate here
            learning_rate = learning_rate  # You can experiment with this value
            optimizer = Adam(learning_rate=learning_rate, clipvalue=clipping_rate if clipping_rate is not None else 0)

            # Compile the model
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Early Stop
            custom_early_stopping = CustomEarlyStopping(monitor_train='accuracy', monitor_val='val_accuracy',
                                                        threshold=accuracy_stop)

            # Train the model
            model.fit(train_generator, epochs=epochs, validation_data=test_generator, verbose=1,
                      callbacks=[custom_early_stopping])

            # Save the model to the specified path
            model.save(model_output)

            print(f"Model saved to {model_output}")

        except Exception as e:
            raise Exception(f"Error building LSTM model for symbol {symbol}: {e}")
    '''
    def train_LSTM_daily_stateful(self, training_series_df, model_output, symbol, classif_key, epochs, variables_csv,
                         timestamps, n_neurons, learning_rate, reg_rate, dropout_rate, clipping_rate=None,
                         accuracy_stop=None, inner_activation='tanh', batch_size=1):
        """
        Train a stateful LSTM model for market classification and save it.
        """
        try:

            # Fill missing values and pre-format
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)
            self.__preformat_training_set__(training_series_df)

            # Get train/test sets
            X_train, X_test, y_train, y_test = self.__get_training_sets__(training_series_df,
                                                                          "trading_symbol", "date",
                                                                          classif_key,
                                                                          variables_csv=variables_csv,
                                                                          test_size=0.2)

            # Trim data to match batch size
            def trim_data(X, y, timesteps, batch_size):
                n_samples = len(X) - timesteps
                n_samples = (n_samples // batch_size) * batch_size
                return X[:n_samples + timesteps], y[:n_samples + timesteps]

            X_train, y_train = trim_data(X_train, y_train, timestamps, batch_size)
            X_test, y_test = trim_data(X_test, y_test, timestamps, batch_size)

            # Create sequences manually
            def create_dataset(X, y, timesteps):
                Xs, ys = [], []
                for i in range(len(X) - timesteps):
                    Xs.append(X[i:i + timesteps])
                    ys.append(y[i + timesteps])
                return np.array(Xs), np.array(ys)

            X_train_seq, y_train_seq = create_dataset(X_train, y_train, timestamps)
            X_test_seq, y_test_seq = create_dataset(X_test, y_test, timestamps)

            # Trim again to match batch size
            def enforce_batch_shape(X, y, batch_size):
                cut = (X.shape[0] // batch_size) * batch_size
                return X[:cut], y[:cut]

            X_train_seq, y_train_seq = enforce_batch_shape(X_train_seq, y_train_seq, batch_size)
            X_test_seq, y_test_seq = enforce_batch_shape(X_test_seq, y_test_seq, batch_size)

            # Force types to prevent OverflowError
            X_train_seq = X_train_seq.astype(np.float32)
            X_test_seq = X_test_seq.astype(np.float32)
            y_train_seq = y_train_seq.astype(np.int32)
            y_test_seq = y_test_seq.astype(np.int32)

            print("X_train_seq shape:", X_train_seq.shape)
            print("y_train_seq shape:", y_train_seq.shape)
            print("X_test_seq shape:", X_test_seq.shape)
            print("y_test_seq shape:", y_test_seq.shape)
            print("batch_size:", batch_size)
            print("epochs:", epochs)

            # Callback to reset states after each epoch
            class ResetStatesCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    self.model.reset_states()

            # Model architecture
            input_shape = (batch_size, timestamps, X_train_seq.shape[2])
            inputs = Input(batch_shape=input_shape)
            x = LSTM(n_neurons, return_sequences=True, stateful=True, activation=inner_activation)(inputs)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            x = LSTM(n_neurons, stateful=True, activation=inner_activation)(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            from tensorflow.python.keras.regularizers import l2
            outputs = Dense(3, activation='softmax', kernel_regularizer=l2(reg_rate))(x)

            model = Model(inputs, outputs)
            model.compile(optimizer=Adam(learning_rate=learning_rate,
                                         clipvalue=clipping_rate if clipping_rate else 0),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            # Train
            model.fit(X_train_seq, y_train_seq,
                      validation_data=(X_test_seq, y_test_seq),
                      batch_size=int(batch_size),
                      epochs=int(epochs),
                      callbacks=[ResetStatesCallback()],
                      shuffle=False)

            # Save model
            model.save(model_output)
            print(f"Model saved to {model_output}")

        except Exception as e:
            import traceback
            raise Exception(f"Fatal error during model training: {traceback.format_exc()}")

    def preload_model(self,model_to_use):
        # Load the saved LSTM model
        model = tensorflow.keras.models.load_model(model_to_use)
        return  model

    def test_cumulative_window_LSTM(self, symbol, test_series_df, model_to_use, timesteps, price_to_use="close",
                                    preloaded_model=None, variables_csv=None):
        """
        Predict the last action of a cumulative sliding window using a non-stateful model.
        This simulates statefulness by growing the input window step by step.

        Parameters:
            symbol (str): Trading symbol
            test_series_df (pd.DataFrame): The current window of test data
            model_to_use (str): Path to saved .keras model
            timesteps (int): Number of timesteps used by the model
            price_to_use (str): Column suffix for price data
            preloaded_model (keras.Model): Optional preloaded model
            variables_csv (str): Ignored in this version

        Returns:
            pd.DataFrame: DataFrame with one row prediction (last minute of window)
        """
        self.__preformat_test_sets__(test_series_df)

        if preloaded_model is None:
            model = tensorflow.keras.models.load_model(model_to_use)
        else:
            model = preloaded_model

        # Use model-based feature detection
        X_test = self.__get_test_sets_from_model__(test_series_df, model)

        # Extract last <timesteps> from cumulative sequence
        timesteps = model.input_shape[1]
        n_features = model.input_shape[2]
        X_input = X_test[-timesteps:]

        # Validate shape
        if X_input.shape[1] != n_features:
            raise ValueError(f"Expected {n_features} features but got {X_input.shape[1]}.")

        # Reshape to (1, timesteps, n_features)
        input_batch = np.expand_dims(X_input, axis=0)
        print("Predicting with input shape:", input_batch.shape)

        # Run prediction
        pred = model.predict(input_batch, batch_size=1)
        action_idx = np.argmax(pred[0])

        # Map prediction to label
        action_labels = {0: "LONG", 1: "SHORT", 2: "FLAT"}
        action = action_labels[action_idx]

        # Extract last timestamp
        date = pd.to_datetime(test_series_df['date'].iloc[-1], unit='s')
        formatted_date = date.strftime(_OUTPUT_DATE_FORMAT)

        # Build result
        result_df = pd.DataFrame({
            'trading_symbol': [symbol],
            'date': [date],
            'formatted_date': [formatted_date],
            'action': [action]
        })

        result_df = self.__add_trading_prices__(
            test_series_df, result_df,
            f"{price_to_use}_{symbol}", [date],
            "trading_symbol_price"
        )

        return result_df

    # def test_stateful_LSTM(self, symbol, test_series_df, model_to_use, timesteps, price_to_use="close",
    #                                   preloaded_model=None):
    #     self.__preformat_test_sets__(test_series_df)
    #
    #     # Prepare the test dataset
    #     X_test = self.__get_test_sets__(test_series_df, symbol_col="trading_symbol", date_col="date")
    #
    #     # Load the LSTM model
    #     if preloaded_model is None:
    #         model = tensorflow.keras.models.load_model(model_to_use)
    #     else:
    #         model = preloaded_model
    #
    #     # Create a time series generator for sequential test data
    #     test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(
    #         X_test, np.zeros(len(X_test)), length=timesteps, batch_size=1
    #     )
    #
    #     # Perform predictions iteratively
    #     predictions = []
    #     for i in range(len(test_generator)):
    #         X_batch, _ = test_generator[i]  # Get the current batch (1 timestep window)
    #         pred = model.predict(X_batch, batch_size=1)  # No states involved
    #         predictions.append(pred)
    #
    #     # Convert predictions to actions (e.g., LONG, SHORT, FLAT)
    #     predictions = np.vstack(predictions)  # Combine predictions into a single array
    #     actions = np.argmax(predictions, axis=1)
    #
    #     # Map actions to readable labels
    #     action_labels = {0: "LONG", 1: "SHORT", 2: "FLAT"}
    #     action_series = pd.Series(actions).map(action_labels)
    #
    #     # Adjust the DataFrame to match prediction length
    #     dates = pd.to_datetime(test_series_df['date'].iloc[timesteps:].reset_index(drop=True), unit='s')
    #     formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)
    #
    #     # Create the final result DataFrame
    #     result_df = pd.DataFrame({
    #         'trading_symbol': symbol,
    #         'date': dates,
    #         'formatted_date': formatted_dates,
    #         'action': action_series
    #     })
    #
    #     # Add trading prices for better analysis
    #     result_df = self.__add_trading_prices__(test_series_df, result_df, f"{price_to_use}_{symbol}", dates,
    #                                             "trading_symbol_price")
    #
    #     return result_df


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

        print("Resumen del modelo:")
        model.summary()

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