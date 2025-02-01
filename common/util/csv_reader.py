import csv


class CSVReader():

    @staticmethod
    def extract_col_csv(file_path,col):

        col_arr=[]

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)  # Create a CSV reader object
            next(reader)  # Skip the header row

            for row in reader:
                if len(row) > col:  # Ensure the row has at least two columns
                    col_arr.append(row[col])  # Add the symbol (column 2) to the set to ensure uniqueness

        cols_csv = ",".join(sorted(col_arr))  # Convert to CSV string

        return cols_csv