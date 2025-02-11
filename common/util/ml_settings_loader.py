import configparser

class MLSettingsLoader():

    def __init__(self):
        pass


    def load_settings(self,config_file_path):
        try:
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_settings={}


            config_settings["hist_data_conn_str"]= config['DB']['HIST_DATA_CONN_STR']
            config_settings["ml_reports_conn_str"] = config['DB']['ML_REPORTS_CONN_STR']
            config_settings["classification_map_key"] = config['SETTINGS']['CLASSIFICATION_MAP_KEY']
            config_settings["IB_PROD_WS"] = config['SETTINGS']['IB_PROD_WS']
            config_settings["PRIMARY_PROD_WS"] = config['SETTINGS']['PRIMARY_PROD_WS']
            config_settings["IB_DEV_WS"] = config['SETTINGS']['IB_DEV_WS']


            return  config_settings



        except Exception as e:
            raise Exception("Critical error reading config file {}:{}".format(config_file_path,str(e)))