import csv
from typing import List

from fastapi import UploadFile


class CSVReader():

    _DEF_SEP_DELIMETER=";"

    @staticmethod
    async def extract_col_csv_from_content(file_content: list, col: int):
        col_arr = []
        reader = csv.reader(file_content)  # Convertir a lector CSV
        next(reader)  # Saltar la cabecera

        for row in reader:
            if len(row) > col:
                col_arr.append(row[col])

        return ",".join(col_arr)  # Convertir a CSV string

    @staticmethod
    def extract_col_csv_from_file(file,col):
        col_arr = []

        reader = csv.reader(file)  # Create a CSV reader object
        next(reader)  # Skip the header row

        for row in reader:
            if len(row) > col:  # Ensure the row has at least two columns
                col_arr.append(row[col])  # Add the symbol (column 2) to the set to ensure uniqueness

        cols_csv = ",".join(col_arr)  # Convert to CSV string

        return cols_csv

    @staticmethod
    def extract_col_csv(file_path,col,delimeter=_DEF_SEP_DELIMETER):

        col_arr=[]

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimeter)  # Create a CSV reader object
            next(reader)  # Skip the header row

            for row in reader:
                if len(row) > col:  # Ensure the row has at least two columns
                    col_arr.append(row[col])  # Add the symbol (column 2) to the set to ensure uniqueness

        cols_csv = ",".join(col_arr)  # Convert to CSV string

        return cols_csv

    @staticmethod
    def extract_col_arr(file_path, col):
        import csv
        col_arr = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')  # Using ';' as delimiter
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) > col:
                    col_arr.append(row[col])
        return col_arr
