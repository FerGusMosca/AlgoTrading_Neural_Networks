from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

from common.util.dataframe_filler import DataframeFiller
from logic_layer.model_creators.base_model_creator import BaseModelCreator


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
            X_full, _, y_full, _ = self.__get_training_sets__(training_series_df,
                                                              "trading_symbol", "date",
                                                              classif_key,
                                                              variables_csv=variables_csv,
                                                              test_size=0.2)  # Use all data for training

            print("Class distribution in y_full:", np.bincount(y_full))

            # Normalize the feature matrix
            scaler = StandardScaler()
            X_full = scaler.fit_transform(X_full)

            # Use TimeSeriesSplit to respect time dependency in validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_accuracies, val_f1_scores = [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
                print(f"\nTraining fold {fold + 1}/{tscv.n_splits}...")

                X_train, X_val = X_full[train_idx], X_full[val_idx]
                y_train, y_val = y_full[train_idx], y_full[val_idx]

                # Instantiate and train the model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    class_weight=class_weight,
                    random_state=42
                )

                model.fit(X_train, y_train)

                # Predict and evaluate
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

            final_model.fit(X_full, y_full)

            # Save the model to disk
            joblib.dump(final_model, model_output)
            print(f"Random Forest model saved to {model_output}")

        except Exception as e:
            raise Exception(f"Fatal error during Random Forest training: {e}")
