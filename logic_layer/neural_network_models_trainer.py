import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetworkModelTrainer():
    def __init__(self,p_logger):
        self.logger=p_logger

    def __format_x_date(self,X, date_col):
        if date_col is not None:
            # Convert 'date_col' to datetime format
            X[date_col] = pd.to_datetime(X[date_col])  # Convert 'date_col' column to datetime

            # Extract date components
            X['year'] = X[date_col].dt.year  # Extract year from the date
            X['month'] = X[date_col].dt.month  # Extract month from the date
            X['day'] = X[date_col].dt.day  # Extract day from the date
            X['weekday'] = X[date_col].dt.weekday  # Extract weekday from the date

            # Create cyclical features to capture cyclical nature
            X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)  # Sine transformation for month
            X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)  # Cosine transformation for month
            X['weekday_sin'] = np.sin(2 * np.pi * X['weekday'] / 7)  # Sine transformation for weekday
            X['weekday_cos'] = np.cos(2 * np.pi * X['weekday'] / 7)  # Cosine transformation for weekday

            # Convert 'date_col' to a numerical timestamp (optional)
            X['date_timestamp'] = X[date_col].astype(np.int64) // 10**9   # Convert 'date_col' to numerical timestamp (seconds since epoch)
            # Drop the original 'date_col' column as it is no longer needed
            X = X.drop(columns=[date_col])  # Remove the original 'date_col' column
        return X

    def __get_model_variables(self,series_df,classification_col):

        series_df = series_df.dropna(thresh=len(series_df.columns) - 3)
        # Separate independent and dependent variables
        X = series_df.drop(columns=[classification_col])  # Independent variables

        X = self.__format_x_date(X, "date")

        y = series_df[classification_col]  # Dependent variable (classification)
        y = y.map({"LONG": 1, "SHORT": 0})  # Map "LONG" to 1 and "SHORT" to 0
        y = y.astype('int')  # Ensure 'y' is of integer type

        return  X,y

    def train_neural_network(self, series_df, classification_col, depth, learning_rate, epochs, model_output,
                             dropout_rate=0.5, l2_reg=0.0):
        """
            Train a neural network model on the provided data.

            Parameters:
            - series_df: DataFrame containing the training data.
            - classification_col: The column name for the classification target variable.
            - depth: Number of units in each hidden layer and the number of hidden layers.
            - learning_rate: Learning rate for the optimizer.
            - epochs: Number of epochs to train the model.
            - model_output: Path where the trained model will be saved.
            - dropout_rate: Rate for Dropout regularization.
            - l2_reg: L2 regularization factor for the kernel weights.

            Returns:
            - Trained Keras model.
            """

        X,y = self.__get_model_variables(series_df, classification_col)

        # Create a Sequential model
        model = Sequential()

        # Get the number of input features
        input_dim = X.shape[1]

        # Add the first hidden layer
        model.add(Dense(units=depth, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        # Add additional hidden layers
        for _ in range(depth - 1):
            model.add(Dense(units=depth, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
            model.add(Dropout(dropout_rate))

        # Add the output layer
        model.add(Dense(1, activation='sigmoid'))  # For binary classification

        # Define the optimizer
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Train the model
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)

        # Save the trained model
        model.save(model_output.replace('"', ''))

        return model


    def run_predictions(self,test_series_df,classification_col,model_to_use):

        X_test,y=self.__get_model_variables(test_series_df,classification_col)

        model = tf.keras.models.load_model(model_to_use.replace('"',""))

        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary (0 or 1)

        # Convert predictions to "LONG" or "SHORT"
        predictions = ["LONG" if pred == 1 else "SHORT" for pred in predictions]

        # Prepare the output dataframe
        df_Y = pd.DataFrame({'Prediction': predictions})
        preds_df = pd.concat([test_series_df["date"], df_Y['Prediction']], axis=1)

        predictions_dict = {"Neural_Network": preds_df}

        return  predictions_dict
