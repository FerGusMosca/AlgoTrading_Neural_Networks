import pandas as pd


class DataframeConcat:

    def __init__(self):
        pass

    @staticmethod
    def add_row(series_df_1, series_df_2):
        # Check if series_df_2 contains exactly one row
        if series_df_2.shape[0] == 1:
            # Check if series_df_1 is empty
            if series_df_1.empty:
                # If series_df_1 is empty, simply return series_df_2
                return series_df_2.copy()
            else:
                # Concatenate series_df_1 with the single row from series_df_2
                series_df_1 = pd.concat([series_df_1, series_df_2], ignore_index=True)
                return series_df_1
        else:
            raise ValueError("series_df_2 should contain exactly one row.")

    @staticmethod
    def concat_df(series_df_1, series_df_2):
        # Check if series_df_1 is empty
        if series_df_1.empty:
            #print("Do not concatenate: series_df_1 is empty.")
            return series_df_2  # Return the original series_df_1 as it is

        if series_df_2.empty:
            #print("Do not concatenate: series_df_1 is empty.")
            return series_df_1  # Return the original series_df_1 as it is

        # Check if series_df_2 contains only NA or empty values
        if not series_df_2.notna().all().all():
            #print("Do not concatenate: series_df_2 contains only NA or empty values.")
            return  series_df_2
        else:
            # Concatenate only if series_df_2 is not empty or does not contain only NA values
            series_df_1 = pd.concat([series_df_1, series_df_2], ignore_index=True)

        return series_df_1