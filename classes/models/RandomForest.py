from sklearn.ensemble import RandomForestClassifier

from classes.models.Model import Model


class RandomForest(Model):
    def __init__(self):
        super().__init__()
        self.classifier = RandomForestClassifier()
