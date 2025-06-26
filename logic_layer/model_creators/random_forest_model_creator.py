import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib

from common.util.dataframe_filler import DataframeFiller
from common.util.light_logger import LightLogger
from logic_layer.model_creators.base_model_creator import BaseModelCreator

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
class RandomForestModelCreator(BaseModelCreator):




    def __eval_class_distributions__(self,model,label_encoder,probs):
        long_index = None
        if label_encoder is not None:
            class_labels = model.classes_
            long_label = label_encoder.transform(["LONG"])[0]
            if long_label in class_labels:
                long_index = list(class_labels).index(long_label)
                LightLogger.do_log(f"[RF TEST] Class index mapping → LONG={long_index}")
            else:
                LightLogger.do_log("[RF TEST] WARNING: Class 'LONG' not found in model.classes_")
        else:
            long_index = 0  # fallback default

        if long_index is not None and probs.shape[1] > long_index:
            prob_long = probs[:, long_index]
        else:
            prob_long = np.zeros(probs.shape[0])  # assume 0% prob of LONG
            LightLogger.do_log("[RF TEST] WARNING: Missing LONG prob. Defaulting to 0% LONG.")


    def train_random_forest_daily(self, training_series_df, model_output, symbol, classif_key,
                                  series_csv, n_estimators=100, max_depth=None,
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
                variables_csv=series_csv,
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

                importances = model.feature_importances_
                for name, imp in zip(training_series_df.columns, importances):
                    if name not in ['symbol', 'date', classif_key]:
                        LightLogger.do_log(f"[RF TRAIN] Feature importance → {name}: {imp:.5f}")

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
            scaler_output = model_output.replace(".pkl", "_scaler.pkl")
            joblib.dump(scaler, scaler_output)
            print(f"Random Forest model saved to {model_output}")

            return label_encoder

        except Exception as e:
            raise Exception(f"Fatal error during Random Forest training: {e}")

    def test_RF_scalping(self, symbol, test_series_df, model_to_use, price_to_use="close",
                         make_stationary=True, normalize=True, series_csv=None,
                         threshold=0.5, preloaded_model=None,label_encoder=None):
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

        X_test = self.__get_test_sets_with_scaler__(
            df=test_series_df,
            variables_csv=series_csv,
            model_to_use=model_to_use
        )

        if preloaded_model is None:
            model = joblib.load(model_to_use)
        else:
            model = preloaded_model

        probs = model.predict_proba(X_test)
        # Log raw probabilities
        mean_probs = np.mean(probs, axis=0)
        LightLogger.do_log(f"[RF TEST] Mean class probabilities: {mean_probs}")

        # Log the model's class index and class labels
        LightLogger.do_log(f"[RF TEST] Model classes_: {model.classes_}")
        LightLogger.do_log(f"[RF TEST] LabelEncoder classes_: {label_encoder.classes_}")
        self.__eval_class_distributions__(model,label_encoder,probs)
        # Get the index for class "LONG" and "SHORT"
        long_index = label_encoder.transform(["LONG"])[0]
        short_index = label_encoder.transform(["SHORT"])[0]

        LightLogger.do_log(f"[RF TEST] Class index mapping → LONG={long_index}, SHORT={short_index}")
        LightLogger.do_log(f"[RF TEST] Model classes_ → {model.classes_}")

        # Use correct column index from predict_proba
        prob_long = probs[:, long_index]

        # Apply threshold decision rule
        actions = np.where(prob_long >= threshold, long_index, short_index)
        # Log distribution of predicted probabilities
        import matplotlib.pyplot as plt
        plt.hist(prob_long, bins=50)
        plt.title(f"Distribution of LONG probabilities (threshold={threshold})")
        plt.xlabel("Probability of LONG")
        plt.ylabel("Frequency")
        plt.savefig("long_prob_distribution.png")  # O usá plt.show() si estás en un entorno interactivo

        # Convert numeric predictions back to original labels
        action_series = label_encoder.inverse_transform(actions)

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
        LightLogger.do_log(f"[RF TEST] First 5 predictions:\n{result_df[['date', 'Prediction']].head()}")

        return result_df, states

