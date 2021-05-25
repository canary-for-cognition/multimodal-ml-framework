from typing import Dict

from classes.CrossValidator import CrossValidator
from classes.models.Model import Model
from classes.tasks.Task import Task


class SingleTasksCV(CrossValidator):
    def __init__(self):
        super().__init__()
        pass

    def cross_validate(self, model: Model, task: Task) -> Dict:
        pass
