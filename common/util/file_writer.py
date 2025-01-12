import os
import pickle

from common.util.light_logger import LightLogger


class FileWriter():


    @staticmethod
    def __write_file__( file_path, content, mode):
        with open(file_path, mode) as file:
            file.write(content)


    @staticmethod
    def __dump_model__(output_path,file_name,model,y_mapping,mode):
        file_path = f"{output_path}{file_name}.pkl"

        with open(file_path,  mode) as file:
            pickle.dump({'model': model, 'label_mapping': y_mapping}, file)


    @staticmethod
    def __create_directory__(root_path,directory_name):
        full_path = os.path.join(root_path, directory_name)
        if not os.path.exists(full_path):
            LightLogger.do_log(f"Creating new directory {full_path}")
            os.makedirs(full_path)

