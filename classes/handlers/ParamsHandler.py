import yaml
import os


class ParamsHandler:

    @staticmethod
    def load_parameters(filename: str) -> dict:
        with open(os.path.join(os.getcwd(), 'params', filename + '.yaml')) as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_parameters(params: dict, filename: str):
        with open(os.path.join(os.getcwd(), 'params', filename + '.yaml'), 'w') as file:
            yaml.dump(params, file)
