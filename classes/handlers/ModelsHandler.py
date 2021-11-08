from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")


class ModelsHandler:
    def __init__(self):
        # empty init
        pass

    @staticmethod
    def get_models(classifiers: list) -> list:
        models = []

        for model in classifiers:
            if model == 'RandomForest':
                models.append(RandomForestClassifier())
            elif model == 'GausNaiveBayes':
                models.append(GaussianNB())
            elif model == 'LogReg':
                models.append(LogisticRegression())
            elif model == 'dummy':
                models.append(DummyClassifier())
            elif model == 'AdaBoost':
                models.append(AdaBoostClassifier())
            elif model == 'Bagging':
                models.append(BaggingClassifier())
            elif model == 'GradBoost':
                models.append(GradientBoostingClassifier())
            else:
                raise ("invalid classifier: %s", model)
        return models

    @staticmethod
    def get_model(classifier: str) -> object:
        if classifier == 'RandomForest':
            return RandomForestClassifier()
        elif classifier == 'GausNaiveBayes':
            return GaussianNB()
        elif classifier == 'LogReg':
            return LogisticRegression()
        elif classifier == 'dummy':
            return DummyClassifier()
        elif classifier == 'AdaBoost':
            return AdaBoostClassifier()
        elif classifier == 'Bagging':
            return BaggingClassifier()
        elif classifier == 'GradBoost':
            return GradientBoostingClassifier()
        else:
            raise ("invalid classifier: %s", classifier)
