from abc import ABC
from typing import List

""" Abstract class """


class Task(ABC):
    def __init__(self):
        pass

    def get_data(self) -> List:
        pass
