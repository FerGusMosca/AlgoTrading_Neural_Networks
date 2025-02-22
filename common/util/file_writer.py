import os
import pickle
from fastapi import UploadFile, File
from pathlib import Path
from common.util.light_logger import LightLogger


class FileWriter():

    @staticmethod
    async def dump_on_path(file: UploadFile = File(...), folder="./temp"):
        """
        Dumps the uploaded file to a specified folder and returns the new file path.

        Parameters:
            file (UploadFile): The file uploaded via FastAPI.
            folder (str): The target directory where the file should be saved (default is "./temp").

        Returns:
            str: The full path to the saved file.
        """
        # Create the target folder if it does not exist
        target_folder = Path(folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        # Construct the full file path using the original filename
        file_path = target_folder / file.filename

        try:
            # Read the file contents asynchronously
            contents = await file.read()
            # Write the contents to the new file in binary mode
            with open(file_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            # Raise an exception if something goes wrong
            raise e

        # Return the file path as a string
        return str(file_path)

    @staticmethod
    def __write_file__(file_path, content, mode):
        with open(file_path, mode) as file:
            file.write(content)

    @staticmethod
    def __dump_model__(output_path, file_name, model, y_mapping, mode):
        file_path = f"{output_path}{file_name}.pkl"

        with open(file_path, mode) as file:
            pickle.dump({'model': model, 'label_mapping': y_mapping}, file)

    @staticmethod
    def __create_directory__(root_path, directory_name):
        full_path = os.path.join(root_path, directory_name)
        if not os.path.exists(full_path):
            LightLogger.do_log(f"Creating new directory {full_path}")
            os.makedirs(full_path)
