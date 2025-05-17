import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib

from common.util.dataframe_filler import DataframeFiller
from logic_layer.model_creators.base_model_creator import BaseModelCreator

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
class RandomForestModelCreator(BaseModelCreator):
    def train_random_forest_daily(self, training_series_df, model_output, symbol, classif_key,
                                  variables_csv, n_estimators=100, max_depth=None,
                                  min_samples_split=2, criterion='gini', make_stationary=False,
                                  class_weight=None):
        try:
            # Fill missing values in the dataset
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)

            # Make the dataset stationary if needed
            if make_stationary:
                training_series_df = self.__make_stationary__(training_series_df)

            # Preformat the dataset (e.g., encode symbols, parse dates, etc.)
            self.__preformat_training_set__(training_series_df)

            # Extract training features and labels
            X_train, X_test, y_train, y_test, label_encoder, scaler = self.__get_training_sets__(
                training_series_df,
                symbol_col="trading_symbol",
                date_col="date",
                classif_key=classif_key,
                variables_csv=variables_csv,
                test_size=0.2,
                return_encoder_and_scaler=True
            )

            print("Class distribution in y_train:", np.bincount(y_train))

            # Use TimeSeriesSplit to respect time dependency in validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_accuracies, val_f1_scores = [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                print(f"\nTraining fold {fold + 1}/{tscv.n_splits}...")

                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    class_weight=class_weight,
                    random_state=42
                )
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')

                print(f"Fold {fold + 1} - Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")
                print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

                val_accuracies.append(acc)
                val_f1_scores.append(f1)

            # Print overall performance metrics
            print(f"\nAverage accuracy: {np.mean(val_accuracies):.4f}")
            print(f"Average F1 Score: {np.mean(val_f1_scores):.4f}")

            # Train the final model on the entire dataset
            final_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42
            )
            final_model.fit(X_train, y_train)

            print("Labels in y_train (raw):", y_train)
            print("Unique encoded labels:", np.unique(y_train))

            # Save the model to disk
            joblib.dump(final_model, model_output)
            print(f"Random Forest model saved to {model_output}")

        except Exception as e:
            raise Exception(f"Fatal error during Random Forest training: {e}")

    def test_RF_scalping(self, symbol, test_series_df, model_to_use, price_to_use="close",
                         make_stationary=True, normalize=True, variables_csv=None,
                         threshold=0.5, preloaded_model=None):
        """
        Test a Random Forest model on given data and return predicted actions.

        Parameters:
        - symbol: string, trading symbol.
        - test_series_df: pd.DataFrame, dataset with features and target.
        - model_to_use: path to saved RF model (.pkl) or preloaded model.
        - price_to_use: column base for trading price ('close', 'open', etc.).
        - make_stationary: bool, whether to difference the data.
        - normalize: bool, whether to scale the data using stored scaler.
        - variables_csv: comma-separated string of features to use.
        - threshold: float, probability threshold for LONG classification.
        - preloaded_model: optional, already loaded model to skip disk I/O.

        Returns:
        - result_df: pd.DataFrame with trading decisions.
        - states: dict (empty, for compatibility).
        """
        if make_stationary:
            test_series_df, _ = self.__make_stationary_with_memory__(test_series_df)

        self.__preformat_test_sets__(test_series_df)

        X_test = self.__get_test_sets__(test_series_df, symbol_col="trading_symbol", date_col="date",
                                        variables_csv=variables_csv, normalize=normalize)

        if preloaded_model is None:
            model = joblib.load(model_to_use)
        else:
            model = preloaded_model

        probs = model.predict_proba(X_test)
        prob_long = probs[:, 0]  # Assuming class 0 = LONG
        actions = np.where(prob_long >= threshold, 0, 1)  # 0=LONG, 1=SHORT
        action_labels = {0: "LONG", 1: "SHORT"}
        action_series = pd.Series(actions).map(action_labels)

        # Align dates (no timesteps shift needed)
        dates = pd.to_datetime(test_series_df['date'].reset_index(drop=True), unit='s')
        formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)

        result_df = pd.DataFrame({
            'trading_symbol': symbol,
            'date': dates,
            'formatted_date': formatted_dates,
            'Prediction': action_series
        })

        test_series_df = test_series_df.rename(columns={
            "open": f"open_{symbol}",
            "high": f"high_{symbol}",
            "low": f"low_{symbol}",
            "close": f"close_{symbol}",
            "symbol": "trading_symbol"
        })

        result_df = self.__add_trading_prices__(test_series_df, result_df, f"{price_to_use}_{symbol}", dates,
                                                "trading_symbol_price")

        states = {}  # no states for RF
        return result_df, states

