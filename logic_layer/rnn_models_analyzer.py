import pandas as pd
import numpy as np
import tensorflow
from keras.src.layers import TimeDistributed
from keras.src.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class RNNModelsAnalyzer:

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
    def __preformat_training_set__(self,training_series_df):

        training_series_df.dropna(inplace=True)

    def __get_test_sets__(self, test_series_df, symbol_col='trading_symbol', date_col='date'):
        """
        Prepare the test set from the given DataFrame.

        Parameters:
        test_series_df (pd.DataFrame): DataFrame containing the time series test data.
        symbol_col (str): The column name for the trading symbol.
        date_col (str): The column name for the date.
        feature_columns (list of str): List of feature columns to be used, should be the same as in training.

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


    def __get_training_sets__(self,training_series_df,symbol_col='trading_symbol',date_col='date',
                              classif_key="classif_col"):
        """
            Prepare the training and test sets from the given DataFrame.

            Parameters:
            training_series_df (pd.DataFrame): DataFrame containing the time series data.
            classif_key (str): The column name of the classification target.
            safety_minutes (int): Number of time steps to look back in the time series.

            Returns:
            tuple: X_train, X_test, y_train, y_test
            """
        # Preprocess 'trading_symbol' column (convert to numeric using LabelEncoder)
        if symbol_col in training_series_df.columns:
            label_encoder_symbol = LabelEncoder()
            training_series_df[symbol_col] = label_encoder_symbol.fit_transform(
                training_series_df[symbol_col])

        # Preprocess 'date' column (convert to timestamp)
        if date_col in training_series_df.columns:
            training_series_df[date_col] = pd.to_datetime(training_series_df[date_col])  # Ensure it's a datetime object
            training_series_df[date_col] = training_series_df[date_col].map(pd.Timestamp.timestamp)  # Convert to timestamp

        # Extract feature columns (all columns except the classification column)
        feature_columns = [col for col in training_series_df.columns if col != classif_key]
        X = training_series_df[feature_columns].values

        # Extract target variable (classification column)
        y = training_series_df[classif_key].values

        # Normalize the feature data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Encode the target variable (classif_key) to numeric values 1, 2, 3
        label_encoder_target = LabelEncoder()
        y_encoded = label_encoder_target.fit_transform(y)  # 0, 1, 2

        # Create a time series generator for training data
        #generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X, y_encoded, length=safety_minutes, batch_size=1)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=False)

        return X_train, X_test, y_train, y_test

    #endregion


    #region Public Methods

    def build_LSTM(self, training_series_df, model_output, symbol, classif_key, epochs,
                   timestamps, n_neurons, learning_rate, reg_rate, dropout_rate):
        """
        Build and train an LSTM model on the given training data and save the model.

        Parameters:
        training_series_df (pd.DataFrame): DataFrame containing the training data.
        model_output (str): Path where the trained model will be saved.
        classif_key (str): The column name of the classification target.
        safety_minutes (int): Number of time steps to look back in the time series.
        """
        try:

            self.__preformat_training_set__(training_series_df)
            # Get training and test sets
            X_train, X_test, y_train, y_test = self.__get_training_sets__(training_series_df,
                                                                          "trading_symbol", "date",
                                                                          classif_key
                                                                          )

            print("X_Train: NaN={} Inf={}".format(np.isnan(X_train).sum(), np.isinf(X_train).sum()))

            # number of timestamps to use
            timesteps = timestamps

            # Generador de series temporales para datos de entrenamiento y prueba
            train_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_train, y_train,
                                                                                          length=timesteps,
                                                                                          batch_size=1)
            test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_test, y_test,
                                                                                         length=timesteps, batch_size=1)

            # Define the LSTM model
            model = tensorflow.keras.models.Sequential()
            model.add(
                LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(timesteps, X_train.shape[1])))
            model.add(Dropout(dropout_rate))  # Dropout layer with 20% dropout rate
            model.add(LSTM(n_neurons, activation='relu'))  # Another LSTM layer without return_sequences
            model.add(Dropout(dropout_rate))  # Dropout layer with 20% dropout rate
            model.add(Dense(3, activation='softmax',
                            kernel_regularizer=tensorflow.keras.regularizers.l2(
                                reg_rate)))  # Three classes: LONG, SHORT, FLAT

            # Adjust the learning rate here
            learning_rate = learning_rate  # You can experiment with this value
            optimizer = Adam(learning_rate=learning_rate)

            # Compile the model
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(train_generator, epochs=epochs, validation_data=test_generator, verbose=1)

            # Save the model to the specified path
            model.save(model_output)

            print(f"Model saved to {model_output}")

        except Exception as e:
            raise Exception(f"Error building LSTM model for symbol {symbol}: {e}")



    def test_LSTM(self,symbol,test_series_df, model_to_use,timestamps):

        self.__preformat_test_sets__(test_series_df)

        X_test=self.__get_test_sets__(test_series_df,symbol_col="trading_symbol",date_col="date")

        # Load the saved LSTM model
        model = tensorflow.keras.models.load_model(model_to_use)

        # timestamps= Number of timestamps used in training (adjust this to match your model's training configuration)
        # Create a time series generator for the test data
        test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(
            X_test, np.zeros(len(X_test)), length=timestamps, batch_size=1
        )

        # Generate predictions
        predictions = model.predict(test_generator)

        # Convert predictions to actions (LONG, SHORT, FLAT)
        actions = np.argmax(predictions, axis=1)
        action_labels = {0: "LONG", 1: "SHORT", 2: "FLAT"}
        action_series = pd.Series(actions).map(action_labels)

        # Adjust the DataFrame to match the length of the predictions
        dates = pd.to_datetime(test_series_df['date'].iloc[timestamps:].reset_index(drop=True), unit='s')
        formatted_dates = dates.dt.strftime('%m/%d/%Y %H:%M:%S')
        symbols = test_series_df['trading_symbol'].iloc[timestamps:].reset_index(drop=True)

        # Create the final output DataFrame
        result_df = pd.DataFrame({
            'trading_symbol': symbols,
            'date': dates,
            'formatted_date': formatted_dates,
            'action': action_series
        })


        result_df=self.__add_trading_prices__(test_series_df,result_df,symbol,dates,"trading_symbol_price")

        return result_df

    #endregion