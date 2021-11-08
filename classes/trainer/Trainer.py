from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.ModelsHandler import ModelsHandler

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
"""
Abstract class Trainer
"""


class Trainer:
    def __init__(self):
        params = ParamsHandler.load_parameters('settings')
        self.mode = params['mode']
        self.clfs = params['classifiers']
        self.clf = None
        self.splits = None
        self.data = None
        self.x = None
        self.y = None
        self.labels = None
        self.feature_set = None
        self.seed = None

        self.models = []
        self.fold_preds_train = []
        self.fold_pred_probs_train = []
        self.fold_preds_test = []
        self.fold_pred_probs_test = []

        self.x_train_fs = []
        self.x_test_fs = []
        self.y_train = []
        self.y_test = []
        self.meta_clf = None
        self.aggregation_method = params['aggregation_method']

        self.preds = {}
        self.pred_probs = {}
        self.results = {}
        self.method = None
        self.k_range = None
        self.best_k = {}
        self.best_score = {}
        self.feature_scores_fold = {}
        self.feature_scores_all = {}

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True) -> object:
        """
        (abstract) train -> function used for training a given classifier with the data
        :param data: the data to use for training. data usually contains x, y, labels as keys
        :param clf: which classifier to use for training.
        :param feature_set: the name of features (columns of x)
        :param feature_importance: boolean that decides whether feature importance code should run or not
        :param seed: the random seed for training
        :return: trainer object
        """
        pass

    @staticmethod
    def calculate_task_fusion_results(data):
        """
        (abstract) calculate_task_fusion_results -> function to recalculate metrics after averaging (only called for fusion)
        :param data: data after averaging
        :return: object with recalculated metrics
        """
        pass

    @staticmethod
    def average_results(data: list, model) -> object:
        """
        :param data: list of Trainer objects that contain attributes pred_probs, preds, etc.
        :param model: classifier for which the aggregation is to be done (only used to refer to a particular entry in the dictionary)
        :return: Trainer object with updated values
        """

        method = 'task_fusion'
        avg_preds = {}
        avg_pred_probs = {}

        sub_data = None
        num = 0
        new_data = None

        # this portion gets activated when across_tasks or across modalities aggregation is required
        # since the model being passed is a single model (either GNB, or RF, or LR)
        if type(model) == str:
            new_data = data[-1][model]
            num = len(data)
            sub_data = np.array([data[t][model] for t in range(num)])

        # this portion gets activated when within_tasks aggregation is required
        # since the models being passed will be more than one
        elif type(model) == list:
            new_data = data[model[-1]]
            num = len(model)
            sub_data = np.array([data[m] for m in model])

        # sub_data will hold all the DementiaCV instances for a particular model, across all tasks
        # so for task='PupilCalib+CookieTheft+Reading+Memory':
        #        sub_data[0] = DementiaCV class for PupilCalib, some model
        #        sub_data[1] = DementiaCV class for CookieTheft, some model.. so on.

        # pred_probs --------------------------------------------------------------------------------------------------

        # find the union of all pids across all tasks
        union_pids = np.unique(np.concatenate([list(sub_data[i].pred_probs[method].keys()) for i in range(num)]))
        pred_probs_dict = {}

        # averaging the pred_probs for a certain PID whenever it's seen across all tasks
        for i in union_pids:
            pred_probs_sum_list = np.zeros(3)
            for t in range(num):
                if i in sub_data[t].pred_probs[method]:
                    pred_probs_sum_list[0] += sub_data[t].pred_probs[method][i][0]
                    pred_probs_sum_list[1] += sub_data[t].pred_probs[method][i][1]
                    pred_probs_sum_list[2] += 1
            pred_probs_dict[i] = np.array([pred_probs_sum_list[0] / pred_probs_sum_list[2], pred_probs_sum_list[1] / pred_probs_sum_list[2]])

        avg_pred_probs[method] = pred_probs_dict
        new_data.pred_probs = avg_pred_probs

        # preds ------------------------------------------------------------------------------------------------------

        # assigning True or False for preds based on what the averaged pred_probs were found in the previous step
        preds_dict = {}
        for i in avg_pred_probs[method]:
            preds_dict[i] = avg_pred_probs[method][i][0] < avg_pred_probs[method][i][1]

        avg_preds[method] = preds_dict
        new_data.preds = avg_preds

        # returned the updated new_data - only pred_probs and preds are changed, the rest are the same as the initially chosen new_data
        return new_data

    @staticmethod
    def compute_save_results(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                             acc_saved=None, fms_saved=None, roc_saved=None,
                             precision_saved=None, recall_saved=None, spec_saved=None):
        """
        compute save results -> function for computing the metrics and saving them
        :param y_true:
        :param y_pred:
        :param y_prob:
        :param acc_saved:
        :param fms_saved:
        :param roc_saved:
        :param precision_saved:
        :param recall_saved:
        :param spec_saved:
        :return:
        """
        if precision_saved is None:
            precision_saved = []
        if spec_saved is None:
            spec_saved = []
        if recall_saved is None:
            recall_saved = []
        if roc_saved is None:
            roc_saved = []
        if fms_saved is None:
            fms_saved = []
        if acc_saved is None:
            acc_saved = []

        # calculating metrics using SKLearn and storing them in lists
        acc_saved.append(accuracy_score(y_true, y_pred))
        fms_saved.append(f1_score(y_true, y_pred))
        roc_saved.append(roc_auc_score(y_true, y_prob))
        precision_saved.append(precision_score(y_true, y_pred))
        recall_saved.append(recall_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec_saved.append(tn / (tn + fp))

        return acc_saved, fms_saved, roc_saved, precision_saved, recall_saved, spec_saved

    def save_results(self, method: str = 'default', acc=None, fms=None, roc=None,
                     precision=None, recall=None, specificity=None,
                     pred=None, pred_prob=None, k_range=None):
        """
        save results -> function for saving results/metrics calculated before into the trainer object's attributes
        :param method:
        :param acc:
        :param fms:
        :param roc:
        :param precision:
        :param recall:
        :param specificity:
        :param pred:
        :param pred_prob:
        :param k_range:
        :return: nothing, saves all the results in the Trainer class object
        """
        self.results[method] = {"acc": np.asarray(acc),
                                "fms": np.asarray(fms),
                                "roc": np.asarray(roc),
                                "precision": np.asarray(precision),
                                "recall": np.asarray(recall),
                                "specificity": np.asarray(specificity)
                                }

        self.best_k[method] = {
            "acc": np.array(k_range)[np.argmax(np.nanmean(acc, axis=0))],
            "fms": np.array(k_range)[np.argmax(np.nanmean(fms, axis=0))],
            "roc": np.array(k_range)[np.argmax(np.nanmean(roc, axis=0))],
            "precision": np.array(k_range)[np.argmax(np.nanmean(precision, axis=0))],
            "recall": np.array(k_range)[np.argmax(np.nanmean(recall, axis=0))],
            "specificity": np.array(k_range)[np.argmax(np.nanmean(specificity, axis=0))],
            "k_range": k_range}

        self.best_score[method] = {"acc": np.max(np.nanmean(acc, axis=0)),
                                   "fms": np.max(np.nanmean(fms, axis=0)),
                                   "roc": np.max(np.nanmean(roc, axis=0)),
                                   "precision": np.max(np.nanmean(precision, axis=0)),
                                   "recall": np.max(np.nanmean(recall, axis=0)),
                                   "specificity": np.max(np.nanmean(specificity, axis=0)),
                                   }
        self.preds[method] = pred
        self.pred_probs[method] = pred_prob
