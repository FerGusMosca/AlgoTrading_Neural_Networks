class CSVWriter():


    @staticmethod
    def write_to_csv(df,output_dir):

        df.to_csv(filepath, index=False)