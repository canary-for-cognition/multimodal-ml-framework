from abc import ABC
from typing import Dict, Union, List

from classes.models.Model import Model
from classes.tasks.Task import Task


class CrossValidator(ABC):
    def __init__(self):
        pass

    def cross_validate(self, m: Union[Model, List], t: Union[Task, List]) -> Dict:
        pass
