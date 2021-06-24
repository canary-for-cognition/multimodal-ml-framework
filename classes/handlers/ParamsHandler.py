from typing import Dict
import yaml


class ParamsHandler:
    def __init__(self):
        pass

    @staticmethod
    def load_parameters(filename: str) -> Dict:
        with open('./params/' + filename + '.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        return config
