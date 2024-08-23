import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class RNNModelsAnalyzer:

    def __init__(self):
        pass


    #region Private Methods


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

    def build_LSTM(self, training_series_df, model_output,symbol, classif_key, safety_minutes):
        """
        Build and train an LSTM model on the given training data and save the model.

        Parameters:
        training_series_df (pd.DataFrame): DataFrame containing the training data.
        model_output (str): Path where the trained model will be saved.
        classif_key (str): The column name of the classification target.
        safety_minutes (int): Number of time steps to look back in the time series.
        """
        try:
            # Get training and test sets
            X_train, X_test, y_train, y_test = self.__get_training_sets__(training_series_df,
                                                                          "trading_symbol","date",
                                                                          classif_key
                                                                          )

            # Número de timesteps que usaremos
            timesteps = 1

            # Generador de series temporales para datos de entrenamiento y prueba
            train_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_train, y_train, length=timesteps, batch_size=1)
            test_generator = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator(X_test, y_test, length=timesteps, batch_size=1)

            # Define the LSTM model
            model = tensorflow.keras.models.Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(timesteps, X_train.shape[1])))
            model.add(Dense(3, activation='softmax'))  # Three classes: LONG, SHORT, FLAT

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(train_generator, epochs=10, validation_data=test_generator, verbose=1)

            # Save the model to the specified path
            model.save(model_output)

            print(f"Model saved to {model_output}")

        except Exception as e:
            raise Exception(f"Error building LSTM model for symbol {symbol}: {e}")


    #endregion