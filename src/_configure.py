import os, sys
import json

class Json_configurer(object):
    def __init__(self, config_file_path):
        assert os.path.exists(config_file_path)

        with open(config_file_path, 'r') as f:
            config_dict = json.load(f)
            f.close()

        self.config_dict = config_dict

    def _getattr_(self, attr):
        if attr in dir(attr):
            return self.__getattribute__(attr)
        else:
            try:
                return self.self.config_dict[attr]
            except KeyError:
                return None
        
