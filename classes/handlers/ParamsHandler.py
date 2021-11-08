import yaml
import os


class ParamsHandler:
    def __init__(self):
        # not required
        pass

    @staticmethod
    def load_parameters(filename: str) -> dict:
        with open(os.path.join(os.getcwd(), 'params', filename + '.yaml')) as file:
            config = yaml.safe_load(file)

        return config

    @staticmethod
    def save_parameters(params: dict, filename: str):
        with open(os.path.join(os.getcwd(), 'params', filename + '.yaml'), 'w') as file:
            yaml.dump(params, file)

