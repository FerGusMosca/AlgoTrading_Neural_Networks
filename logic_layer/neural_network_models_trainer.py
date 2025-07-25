import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from framework.common.logger.message_type import MessageType


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

        #series_df = series_df.dropna(thresh=len(series_df.columns) - 3)

        # Separate independent and dependent variables
        X = series_df.drop(columns=[classification_col])  # Independent variables

        X = self.__format_x_date(X, "date")

        y = series_df[classification_col]  # Dependent variable (classification)
        y = y.map({"LONG": 1, "SHORT": 0})  # Map "LONG" to 1 and "SHORT" to 0
        y = y.astype('int')  # Ensure 'y' is of integer type

        return  X,y

    def __normalize(self, series_df, variables_csv):
        # Convert the comma-separated string of column names to a list
        variables_list = variables_csv.split(',')

        # Separate numeric and non-numeric columns
        numeric_columns = [col for col in variables_list if col in series_df.columns]
        non_numeric_columns = [col for col in series_df.columns if col not in variables_list]

        # Filter the numeric columns to be normalized
        series_df_filtered = series_df[numeric_columns]

        # Convert the filtered DataFrame to a TensorFlow tensor
        tensor_df = tf.convert_to_tensor(series_df_filtered, dtype=tf.float32)

        # Create the normalizer and adapt it to the data
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(tensor_df)

        # Normalize the data
        tensor_normalized = normalizer(tensor_df)

        # Convert the normalized tensor back to a DataFrame
        df_normalized_numeric = pd.DataFrame(tensor_normalized.numpy(), columns=numeric_columns)

        # Add back the non-numeric columns to the normalized DataFrame
        df_normalized = pd.concat(
            [series_df[non_numeric_columns].reset_index(drop=True), df_normalized_numeric.reset_index(drop=True)],
            axis=1)

        return df_normalized

    def __training_set_analysis(self, series_df, classification_col, prefix="Pre depuration balance:"):
        """
        Analyzes the dataset and calculates the percentage distribution of each category in the specified column.

        Args:
        - series_df (pd.DataFrame): DataFrame containing the classification column.
        - classification_col (str): Name of the classification column in the DataFrame.

        Logs:
        - Percentage distribution of each category in the specified column.
        """
        # Check if the classification column exists in the DataFrame
        if classification_col not in series_df.columns:
            raise ValueError(f"The column '{classification_col}' does not exist in the DataFrame.")

        # Count occurrences of each category
        counts = series_df[classification_col].value_counts()

        # Calculate percentage distribution
        percentages = counts / len(series_df) * 100

        # Log results
        self.logger.do_log(f"{prefix} Category distribution in '{classification_col}':", MessageType.INFO)
        for category, percentage in percentages.items():
            self.logger.do_log(f"{category}: {percentage:.2f}%", MessageType.INFO)

    def __training_set_depuration(self, series_df, classification_col, target_distribution=0.5):
        """
        Balances the dataset by depurating (removing) records from the majority class to achieve a target distribution.

        Args:
        - series_df (pd.DataFrame): DataFrame containing the classification column.
        - classification_col (str): Name of the classification column in the DataFrame.
        - target_distribution (float): Desired proportion of the minority class (default is 0.5 for 50%/50%).

        Returns:
        - pd.DataFrame: The depurated DataFrame with balanced class distribution.

        Raises:
        - ValueError: If the classification column does not exist or if the target distribution is not between 0 and 1.
        """
        # Validate target distribution
        if not (0 < target_distribution < 1):
            raise ValueError("Target distribution must be between 0 and 1.")

        # Check if the classification column exists in the DataFrame
        if classification_col not in series_df.columns:
            raise ValueError(f"The column '{classification_col}' does not exist in the DataFrame.")

        # Count occurrences of each category
        counts = series_df[classification_col].value_counts()

        if len(counts) < 2:
            raise ValueError("There must be at least two classes in the classification column.")

        # Determine the majority and minority classes
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()

        # Create separate DataFrames for each class
        majority_df = series_df[series_df[classification_col] == majority_class]
        minority_df = series_df[series_df[classification_col] == minority_class]

        # Determine the number of records to keep for the majority class to match the minority class
        minority_count = len(minority_df)

        # Sample the majority class to match the minority class count
        if len(majority_df) > minority_count:
            majority_df = majority_df.sample(minority_count, random_state=42)

        # Concatenate the balanced DataFrames
        balanced_df = pd.concat([majority_df, minority_df])

        # Shuffle the DataFrame to mix the samples
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return balanced_df

    def train_logistic_regression(self,X, y, learning_rate=0.01, epochs=100, model_output="logistic_regression_model.h5"):
        # Create a Sequential model
        model = Sequential()

        # Add a single dense layer with a sigmoid activation
        model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))

        # Define the optimizer
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)

        # Save the trained model
        model.save(model_output)

        return model

    def train_neural_network(self, series_df,variables_csv, classification_col, depth, learning_rate, epochs, model_output,
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

        series_df=DataframeFiller.fill_missing_values(series_df)

        self.__training_set_analysis(series_df,classification_col)
        #series_df=self.__training_set_depuration(series_df,classification_col)
        #self.__training_set_analysis(series_df, classification_col,"Post depuration balance")

        series_df_norm=self.__normalize(series_df,variables_csv)

        X,y = self.__get_model_variables(series_df_norm, classification_col)

        if depth==1: #We use a simple Log Reggr.
            return self.train_logistic_regression(X,y,learning_rate,epochs,model_output)

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
