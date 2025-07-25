import pandas as pd
class DataframePrinter:

    def __init__(self):
        pass

    @staticmethod
    def print_dataframe(df,filter_col, filter_value, rows):
        # Filter the rows where the 'symbol' column is equal to "SPY" (or any other value)
        filtered_df = df[df[filter_col] == filter_value]

        # Show the first 15 rows, making sure all columns are displayed
        pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

        print(filtered_df.head(rows))

    @staticmethod
    def print_dataframe_head_values(df,filter_col, filter_values_csv, row_number):
        filter_values = filter_values_csv.split(',')

        # Recorrer cada valor en la lista
        for symbol in filter_values:
            print(f"---First {row_number} rows for symbol {symbol}")
            DataframePrinter.print_dataframe(df, filter_col, symbol, row_number)


    @staticmethod
    def print_dataframe_head_values_w_time(df,filter_col, filter_values_csv, row_number,timestamp_col, min_timestamp_val):

        filter_time = pd.to_datetime(min_timestamp_val).time()

        df = df[df[timestamp_col].dt.time >= filter_time]

        filter_values = filter_values_csv.split(',')

        # Recorrer cada valor en la lista
        for symbol in filter_values:
            print(f"---First {row_number} rows for symbol {symbol}")
            DataframePrinter.print_dataframe(df, filter_col, symbol, row_number)

    @staticmethod
    def print_data_farme_head(df,n_rows):
        # Show the first 15 rows, making sure all columns are displayed
        pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

        print(df.head(n_rows))