from datetime import timedelta

from common.util.dataframe_concat import DataframeConcat
from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
import pandas as pd

from framework.common.logger.message_type import MessageType


class DataSetBuilder():


    _1_DAY_INTERVAL="1 day"
    _1_MIN_INTERVAL = "1 min"
    _CLASSIFICATION_COL="classification"

    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):
        self.classification_map_values = []
        self.economic_series_mgr= EconomicSeriesManager(hist_data_conn_str)
        self.date_range_classification_mgr = DateRangeClassificationManager(ml_reports_conn_str)
        self.classification_map_key = p_classification_map_key
        self.logger = logger


    #region Private Methods

    def get_classification_for_date(self,date, timestamp_range_clasifs,not_found_clasif):
        for date_range in timestamp_range_clasifs:
            if date_range.date_start <= date <= date_range.date_end:
                #print(f"Date {date} falls within {date_range.date_start} and {date_range.date_end}, classified as {date_range.classification}")
                return date_range.classification

        #print(f"Date {date} classified as {not_found_clasif}")
        return not_found_clasif

    #endregion

    def get_extreme_dates(self,series_data_dict):
        min_date=None
        max_date=None


        for economic_value_list in series_data_dict.values():
            for econ_value in  economic_value_list:
                if min_date is None or min_date>econ_value.date:
                    min_date=econ_value.date

                if max_date is None or max_date<econ_value.date:
                    max_date=econ_value.date

        return  min_date,max_date

    def eval_all_values_none(self,curr_date_dict):
        not_none=False

        for value in curr_date_dict.values():
            if value is not None:
                not_none=True

        return not not_none

    def build_empty_dataframe(self,series_data_dict):

        column_list=["date"]

        for key in series_data_dict.keys():
            column_list.append(key)

        df = pd.DataFrame(columns=column_list)

        return df

    def assign_classification(self,curr_date):
        classification_value = next((classif_value for classif_value in self.classification_map_values
                                     if classif_value.date_start.date() <= curr_date.date() <= classif_value.date_end.date()), None)

        if classification_value is None:
            raise Exception("Could not find the proper classification for date {}".format(curr_date.date()))

        return  classification_value.classification

    def load_classification_map_date_ranges(self):
        self.classification_map_values=self.date_range_classification_mgr.get_date_range_classification_values(self.classification_map_key)
        pass

    def fill_dataframe(self,series_df,min_date,max_date,series_data_dict,add_classif_col=True):
        curr_date=min_date
        self.logger.do_log("Building the input dataframe from {} to {}".format(min_date.date(),max_date.date()), MessageType.INFO)
        while curr_date<=max_date:
            curr_date_dict={}
            for key in series_data_dict.keys():
                series_value=next((x for x in series_data_dict[key] if x.date == curr_date), None)
                if series_value is not None and series_value.close is not None:
                    curr_date_dict[key]=series_value.close
                else:
                    curr_date_dict[key]=None

            if not self.eval_all_values_none(curr_date_dict):
                curr_date_dict["date"]=curr_date

                if add_classif_col:
                    curr_date_dict[DataSetBuilder._CLASSIFICATION_COL]=self.assign_classification(curr_date)

                self.logger.do_log("Adding dataframe row for date {}".format(curr_date.date()),MessageType.INFO)
                #series_df=series_df.append(curr_date_dict, ignore_index=True)
                #series_df = pd.concat([series_df, pd.DataFrame([curr_date_dict])], ignore_index=True)
                #series_df=DataframeConcat.concat_df(series_df,pd.DataFrame([curr_date_dict]))
                series_df=DataframeConcat.add_row(series_df,pd.DataFrame([curr_date_dict]))
            curr_date = curr_date + timedelta(days=1)

        self.logger.do_log("Input dataframe from {} to {} successfully created: {} rows".format(min_date, max_date,len(series_df)), MessageType.INFO)
        return series_df

    def fill_dataframe_from_economic_value_dict(self,series_data_dict,index):

        economic_values=series_data_dict[index]

        data = {
            'symbol': [ev.symbol for ev in economic_values],
            'interval': [ev.interval for ev in economic_values],
            'date': [ev.date for ev in economic_values],
            'open': [ev.open for ev in economic_values],
            'high': [ev.high for ev in economic_values],
            'low': [ev.low for ev in economic_values],
            'close': [ev.close for ev in economic_values],
            'trade': [ev.trade for ev in economic_values],
            'cash_volume': [ev.cash_volume for ev in economic_values],
            'nominal_volume': [ev.nominal_volume for ev in economic_values],
        }

        df = pd.DataFrame(data)

        return  df


    def merge_minute_series(sel,symbol_min_series_df,variables_min_series_df,symbol_col,date_col, symbol):
        # Step 1: Pivot the variables_min_series_df to turn 'symbol_col' values into columns
        variables_pivot_df = variables_min_series_df.pivot(index=date_col, columns=symbol_col, values='open')

        # Step 2: Rename the pivoted columns to avoid confusion
        variables_pivot_df = variables_pivot_df.rename_axis(None, axis=1).reset_index()

        # Step 3: Merge symbol_min_series_df with variables_pivot_df using 'date_col' as the key
        merged_df = pd.merge(symbol_min_series_df, variables_pivot_df, on=date_col, how='outer')

        # Step 4: Rename columns of symbol_min_series_df in the merged dataframe
        merged_df = merged_df.rename(columns={
            symbol_col: 'trading_{}'.format(symbol_col),
            'open': 'open_{}'.format(symbol),
            'high': 'high_{}'.format(symbol),
            'low': 'low_{}'.format(symbol),
            'close': 'close_{}'.format(symbol)
        })

        # Return the final merged dataframe
        return merged_df

    def build_minute_series(self,series_csv,d_from,d_to,interval=None, output_col=None):

        if interval is None:
            interval = DataSetBuilder._1_MIN_INTERVAL

        series_list = series_csv.split(",")

        series_data_dict = {}

        for serieID in series_list:
            economic_values = self.economic_series_mgr.get_economic_values(serieID,interval,d_from, d_to)
            if len(economic_values) == 0:
                return None #maybe we are in a holiday

            series_data_dict[serieID] = economic_values

        min_series_df = self.build_empty_dataframe(series_data_dict)
        for seriesID in series_data_dict.keys():
            series_df = self.fill_dataframe_from_economic_value_dict(series_data_dict,seriesID)
            min_series_df = pd.concat([min_series_df, series_df], ignore_index=True)

            if seriesID in min_series_df.columns:
                min_series_df = min_series_df.drop(columns=[seriesID])

        if output_col is not None:
            min_series_df = min_series_df[output_col]


        return  min_series_df


    def build_minute_series_classification(self,timestamp_range_clasifs,min_series_df,classif_col_name="classif_col",not_found_clasif="None"):
        # For every row in the dataframe
        min_series_df[classif_col_name] = min_series_df['date'].apply(
            lambda x: self.get_classification_for_date(x, timestamp_range_clasifs,not_found_clasif))

        return min_series_df

    def build_daily_series_classification(self,series_csv,d_from,d_to,add_classif_col=True):
        series_list = series_csv.split(",")

        series_data_dict = {}

        for serieID in series_list:
            economic_values = self.economic_series_mgr.get_economic_values(serieID, DataSetBuilder._1_DAY_INTERVAL,
                                                                           d_from, d_to)
            if len(economic_values)==0:
                raise  Exception("No data found for SeriesID {}".format(serieID))

            series_data_dict[serieID] = economic_values

        min_date, max_date = self.get_extreme_dates(series_data_dict)

        series_df = self.build_empty_dataframe(series_data_dict)

        if add_classif_col:

            self.load_classification_map_date_ranges()

        series_df = self.fill_dataframe(series_df, min_date, max_date, series_data_dict,
                                        add_classif_col=add_classif_col)

        return  series_df

    def group_as_mov_avgs(self,symbol_min_series_df, variables_csv, grouping_mov_avg_unit):
        # Convert the column list from CSV to a Python list
        columns = variables_csv.split(",")

        # Create a copy of the original DataFrame to preserve non-moving-average columns
        result_df = symbol_min_series_df.copy()

        # Calculate moving averages for each column in 'variables_csv'
        for col in columns:
            if col in symbol_min_series_df.columns:
                # Calculate simple moving average for the current column
                result_df[col] = symbol_min_series_df[col].rolling(window=grouping_mov_avg_unit).mean()

        # Fill the first 'grouping_mov_avg_unit - 1' rows with None
        result_df.iloc[:grouping_mov_avg_unit - 1] = None

        return result_df