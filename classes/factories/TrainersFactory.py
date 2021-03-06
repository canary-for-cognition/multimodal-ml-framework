from classes.trainers.Trainer import Trainer
from classes.trainers.SingleModelTrainer import SingleModelTrainer
from classes.trainers.TaskFusionTrainer import TaskFusionTrainer
from classes.trainers.ModelEnsembleTrainer import ModelEnsembleTrainer
from classes.trainers.StackingTrainer import StackingTrainer


class TrainersFactory:
    def __init__(self):
        self.__trainers = {
            "single_tasks": SingleModelTrainer,
            "fusion": TaskFusionTrainer,
            "ensemble": ModelEnsembleTrainer,
            "stack": StackingTrainer
        }

    def get(self, mode: str) -> Trainer:
        """
        get -> returns a Trainer class from the given mode
        :param mode: single_tasks, fusion or ensemble. Used to choose the type of Trainer.
        :return: Trainer class
        """
        if mode not in self.__trainers.keys():
            raise ValueError("Trainer '{}' not supported! Supported trainers are: {}"
                             .format(mode, self.__trainers.keys()))

        return self.__trainers[mode]()
