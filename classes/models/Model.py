from abc import ABC
from typing import List, Dict

""" Abstract class """


class Model(ABC):
    def __init__(self):
        pass

    def train(self, data: List) -> Dict:
        pass
