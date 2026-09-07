from pathlib import Path
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import sklearn
from packaging import version
from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.graphs.graph_builder import GraphBuilder
from common.util.logging.light_logger import LightLogger
from logic_layer.model_creators.base_model_creator import BaseModelCreator

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

class XGBoostModelCreator(BaseModelCreator):

    def __build_features_for_inference__(self,
                                         raw_df: pd.DataFrame,
                                         feature_cols: list,
                                         symbol: str,
                                         make_stationary: bool = True,
                                         state: dict | None = None,
                                         model_file_name=None) -> tuple[pd.DataFrame, dict | None]:
        """
        Prepare features for inference so they exactly match the training feature set.

        Steps:
          - Ensure 'date' is datetime and sorted.
          - Optionally make series stationary using __make_stationary_with_memory__.
          - Harmonize OHLC names (open/high/low/close vs open_SYMBOL/...).
          - Keep only ['date'] + feature_cols, in training order.
          - Coerce to numeric, handle Inf/NaN, cast to float32.

        Returns:
          (features_df_ready, updated_state)
        """

        df = raw_df.copy()

        # 0) date handling
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 1) Stationarity (your method, with state)
        updated_state = state
        if make_stationary:
            # Use your differencing method; it mutates and returns (df, state)
            df, updated_state = self.__make_stationary_with_memory__(df, state=state,model_file_name=model_file_name)

        # 2) Harmonize OHLC names (if training expected suffixed or unsuffixed)
        base_ohlc = ["open", "high", "low", "close"]
        # base -> base_SYMBOL
        for base in base_ohlc:
            expected = f"{base}_{symbol}"
            if expected in feature_cols and expected not in df.columns and base in df.columns:
                df.loc[:, expected] = df[base]
        # base_SYMBOL -> base
        for base in base_ohlc:
            expected_no = base
            expected_suf = f"{base}_{symbol}"
            if expected_no in feature_cols and expected_no not in df.columns and expected_suf in df.columns:
                df.loc[:, expected_no] = df[expected_suf]

        # 3) Ensure all training features exist (create missing as NaN)
        for col in feature_cols:
            if col not in df.columns:
                df.loc[:, col] = np.nan

        # 4) Keep only date + features, in training order
        out = pd.concat([df[["date"]], df[feature_cols]], axis=1)

        # 5) Coerce to numeric (only features), then sanitize
        dup = pd.Index(feature_cols)[pd.Index(feature_cols).duplicated()].unique().tolist()
        if dup:
            raise ValueError(f"Duplicate feature columns detected: {dup}")

        out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
        out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Drop rows that still have NaN in any feature
        mask_ok = np.isfinite(out[feature_cols].to_numpy()).all(axis=1)
        out = out.loc[mask_ok].reset_index(drop=True)

        # 6) Final dtype for XGBoost
        out[feature_cols] = out[feature_cols].astype(np.float32)

        return out, updated_state

    def __eval_class_distributions__(self, model, label_encoder, probs):
        long_index = None
        if label_encoder is not None:
            class_labels = model.classes_
            long_label = label_encoder.transform(["LONG"])[0]
            if long_label in class_labels:
                long_index = list(class_labels).index(long_label)
                LightLogger.do_log(f"[XGB TEST] Class index mapping → LONG={long_index}")
            else:
                LightLogger.do_log("[XGB TEST] WARNING: Class 'LONG' not found in model.classes_")
        else:
            long_index = 0  # fallback default

        if long_index is not None and probs.shape[1] > long_index:
            prob_long = probs[:, long_index]
        else:
            prob_long = np.zeros(probs.shape[0])
            LightLogger.do_log("[XGB TEST] WARNING: Missing LONG prob. Defaulting to 0% LONG.")

    def train_xgboost_daily(
            self,
            training_series_df,
            model_output,
            symbol,
            classif_key,
            series_csv,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            make_stationary=False,
            class_weight=None,
            register_model=None
    ):
        """
        Train a raw XGBoost (multi:softprob) model for daily scalping.  Probabilities are NOT calibrated.
        The resulting bundle includes: booster, feature list, label_encoder and scaler.
        """
        training_series_df = DataframeFiller.fill_missing_values(training_series_df)
        if make_stationary:
            training_series_df = self.__make_stationary__(training_series_df,model_output)

        self.__preformat_training_set__(training_series_df)

        X_train, X_test, y_train, y_test, label_encoder, scaler = self.__get_training_sets__(
            training_series_df,
            symbol_col="trading_symbol",
            date_col="date",
            classif_key=classif_key,
            variables_csv=series_csv,
            test_size=0.2,
            return_encoder_and_scaler=True,
        )

        expected_features = series_csv.split(',')

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Save booster + metadata
        booster_n = model.get_booster().num_features()
        if booster_n != len(expected_features) or booster_n != X_train.shape[1]:
            raise RuntimeError("Mismatch between features and booster structure")

        self.__save_xgb_model_bundle__(
            model,
            expected_features,
            label_encoder,
            scaler,
            model_output
        )

        self.__register_xgb_model_in_mlflow__(model_output, scaler, label_encoder, register_model,model)

        return label_encoder

    def test_XGBoost_scalping(
            self,
            symbol_df,
            symbol,
            features_df,
            model_filename,
            bias="LONG",
            draw_statistics=False,
            make_stationary=True,
            lower_percentile_limit=0.4,
            debug=False,
    ):
        """
        Predict scalping signals using a trained (raw) XGBoost booster.
        The ranking is based on the raw LONG probability.
        percentile_threshold ∈ (0,1] is interpreted as: "take the top (1 - p)% of prob_long".
          e.g. 0.4 → top-60%  /  0.8 → top-20%
        Output: DataFrame with date, close, prob_long and Prediction.
        """

        booster, label_encoder, scaler, feature_cols, _xgb_params = self.__load_xgb_model_bundle__(model_filename)

        features_ready, _ = self.__build_features_for_inference__(
            raw_df=features_df,
            feature_cols=feature_cols,
            symbol=symbol,
            make_stationary=make_stationary,
            state=None,
            model_file_name=model_filename
        )

        merged_df = pd.merge(
            features_ready[["date"] + feature_cols],
            symbol_df[["date", "close"]],
            on="date",
            how="inner",
        )

        X = merged_df[feature_cols].to_numpy(dtype=np.float32)

        # raw booster probabilities
        dmat = xgb.DMatrix(X, missing=np.nan, feature_names=feature_cols)
        y_probs = booster.predict(dmat)
        if y_probs.ndim == 1:
            y_probs = np.column_stack([1 - y_probs, y_probs])

        classes = np.array(label_encoder.classes_)
        idx_map = {c: int(np.where(classes == c)[0][0]) for c in classes}
        long_idx = idx_map["LONG"]
        short_idx = idx_map["SHORT"]

        prob_long = y_probs[:, long_idx]

        # percentile-based cut
        cut_value = np.quantile(prob_long, lower_percentile_limit)
        print(f"[CUT] lower_percentile_limit={lower_percentile_limit:.4f} -> probabilidad de corte={cut_value:.6f} "
              f"({cut_value * 100:.2f}%) | casos totales={len(prob_long)} | "
              f"casos LONG={int((prob_long >= cut_value).sum())} "
              f"({(prob_long >= cut_value).mean() * 100:.2f}%)")

        y_pred_ids = np.where(prob_long >= cut_value, long_idx, short_idx)
        y_class = label_encoder.inverse_transform(y_pred_ids)

        dates = pd.to_datetime(merged_df["date"], unit="s").dt.strftime(_OUTPUT_DATE_FORMAT)

        result_df = pd.DataFrame({
            "date": merged_df["date"].values,
            "formatted_date": dates.values,
            "close": merged_df["close"].values,
            "prob_long": prob_long,
            "Prediction": y_class,
            "bias": bias,
        })

        result_df.attrs["cut_value"] = float(cut_value)
        result_df.attrs["lower_percentile_limit"] = float(lower_percentile_limit)

        if draw_statistics:
            GraphBuilder.plot_long_probability_distributions(prob_long, threshold=cut_value)

        return result_df, features_ready









