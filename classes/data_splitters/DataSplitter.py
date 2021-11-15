from abc import abstractmethod

from classes.handlers.ParamsHandler import ParamsHandler


class DataSplitter:
    def __init__(self):
        params = ParamsHandler.load_parameters('settings')
        self.random_seed = None
        self._mode = params['mode']
        self._num_folds = params['folds']

    @abstractmethod
    def make_splits(self, data: dict, seed: int) -> list:
        """
        (abstract) make splits -> function for creating splits with the given data
        :param data: data to split into nfolds
        :param seed: random seed
        :return: list containing different splits
        """
        pass
