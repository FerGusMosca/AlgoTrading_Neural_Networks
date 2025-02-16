class DataframeFiller:


    def __init__(self):
        pass


    @staticmethod
    def fill_missing_values(series_df, col=None):
        """
        Fill missing values in the DataFrame by carrying forward the last available value.
        If no previous value is available, fill with the next available value if it's within 60 days.

        If a NaN value is found with no previous or next value to fill (within 60 days),
        an exception is raised.

        Parameters:
        - series_df: DataFrame containing the time series data with a 'date' column.

        Returns:
        - DataFrame with missing values filled.

        Raises:
        - ValueError: If there is a NaN value with no previous data and the next available data is more than 60 days ahead.
        """

        # Ensure the DataFrame is sorted by date
        series_df = series_df.sort_values(by='date')

        # Loop through each column to handle NaN values
        for column in series_df.columns:
            if column == 'date':
                continue
            if col is not None and col!=column:
                continue

            # Fill forward with the last available value
            series_df[column] = series_df[column].fillna(method='ffill')

            # Find any remaining NaN values
            nan_indices = series_df[series_df[column].isnull()].index

            for idx in nan_indices:
                # Get the date of the NaN value
                nan_date = series_df.loc[idx, 'date']

                # Find the next available value's index
                next_valid_idx = series_df[column].loc[idx:].first_valid_index()

                if next_valid_idx is not None:
                    next_valid_date = series_df.loc[next_valid_idx, 'date']
                    days_diff = (next_valid_date - nan_date).days

                    if days_diff <= 60:
                        # Fill NaN with the next valid value if within 60 days
                        series_df.at[idx, column] = series_df.at[next_valid_idx, column]
                    else:
                        # Raise an error if the next value is more than 60 days away
                        raise ValueError(
                            f"Feature '{column}' has NaN values until the first data which is on date {next_valid_date}")
                else:
                    raise ValueError(
                        f"Feature '{column}' has NaN values with no future data to fill after date {nan_date}")

        return series_df