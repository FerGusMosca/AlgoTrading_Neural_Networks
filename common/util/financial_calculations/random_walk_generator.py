import numpy as np
import pandas as pd


class RandomWalkGenerator():


    def __init__(self):

        pass

    @staticmethod
    def __fill_with_random_walk_values__(symbol, curr_min_df,
                                         max_cut_timestamp, max_df_timestamp, variables_csv):
        """
        Fills the DataFrame with random walk values for missing timestamps
        between max_cut_timestamp and max_df_timestamp.

        Parameters:
        - symbol (str): Trading symbol to use for fake data.
        - curr_min_df (pd.DataFrame): DataFrame containing the current window of data.
        - max_cut_timestamp (datetime): Last timestamp in the current real data.
        - max_df_timestamp (datetime): Maximum timestamp available in the test series.
        - variables_csv (list): List of columns to project from the last real row.

        Returns:
        - pd.DataFrame: Updated DataFrame with real and generated random walk data.
        """
        if max_cut_timestamp < max_df_timestamp:
            # Get the last real row
            last_real_row = curr_min_df.iloc[-1]
            last_date = last_real_row["date"]

            # Create a range of fictitious timestamps
            future_timestamps = pd.date_range(start=last_date + pd.Timedelta(minutes=1),
                                              end=max_df_timestamp,
                                              freq="1min")

            # Initialize the fake data dictionary
            fake_data = {"date": future_timestamps, "trading_symbol": symbol}

            # Generate random walk for 'close' prices if 'close' is in variables_csv
            for var in variables_csv.split(","):
                if var in last_real_row:
                    last_value = last_real_row[var]
                    random_walk = np.random.normal(loc=0, scale=0.01, size=len(future_timestamps))
                    projected_values = last_value + np.cumsum(random_walk)
                    fake_data[var] = projected_values

            # Project additional columns from the last real row
            for column in variables_csv.split(","):
                if  column in last_real_row:
                    fake_data[column] = [last_real_row[column]] * len(future_timestamps)

            # Create the DataFrame for fake data
            fake_data_df = pd.DataFrame(fake_data)

            # Append the fake data to the current DataFrame
            curr_min_df = pd.concat([curr_min_df, fake_data_df], ignore_index=True)

        return curr_min_df

