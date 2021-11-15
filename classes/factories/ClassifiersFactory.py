import warnings

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


class ClassifiersFactory:
    def __init__(self):
        self.__classifiers = {
            "RandomForest": RandomForestClassifier(),
            "GausNaiveBayes": GaussianNB(),
            "LogReg": LogisticRegression(),
            "dummy": DummyClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Bagging": BaggingClassifier(),
            "GradBoost": GradientBoostingClassifier()
        }

    def get_models(self, classifiers: list) -> list:
        models = []
        for classifier in classifiers:
            if classifier not in self.__classifiers.keys():
                raise ValueError("Invalid classifier '{}'! Supported classifiers are: {}"
                                 .format(classifier, self.__classifiers.keys()))
            models.append(self.__classifiers[classifier])
        return models

    def get_model(self, classifier: str) -> object:
        if classifier not in self.__classifiers.keys():
            raise ValueError("Invalid classifier '{}'! Supported classifiers are: {}"
                             .format(classifier, self.__classifiers.keys()))
        return self.__classifiers[classifier]
