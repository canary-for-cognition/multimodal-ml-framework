from classes.trainers.Trainer import Trainer
from classes.cv.FeatureSelector import FeatureSelector
from classes.factories.ClassifiersFactory import ClassifiersFactory
from classes.handlers.ParamsHandler import ParamsHandler
from classes.factories.DataSplitterFactory import DataSplitterFactory

import numpy as np
import random
import os
import pandas as pd


class TaskFusionTrainer(Trainer):
    def __init__(self):
        super().__init__()

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

        # This portion gets activated when across_tasks or across modalities aggregation is required
        # since the model being passed is a single model (either GNB, or RF, or LR)
        if type(model) == str:
            new_data = data[-1][model]
            num = len(data)
            sub_data = np.array([data[t][model] for t in range(num)])

        # This portion gets activated when within_tasks aggregation is required
        # since the models being passed will be more than one
        elif type(model) == list:
            new_data = data[model[-1]]
            num = len(model)
            sub_data = np.array([data[m] for m in model])

        # Find the union of all pids across all tasks
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
            pred_probs_dict[i] = np.array(
                [pred_probs_sum_list[0] / pred_probs_sum_list[2], pred_probs_sum_list[1] / pred_probs_sum_list[2]])

        avg_pred_probs[method] = pred_probs_dict
        new_data.pred_probs = avg_pred_probs

        # preds ------------------------------------------------------------------------------------------------------

        # assigning True or False for preds based on what the averaged pred_probs were found in the previous step
        preds_dict = {}
        for i in avg_pred_probs[method]:
            preds_dict[i] = avg_pred_probs[method][i][0] < avg_pred_probs[method][i][1]

        avg_preds[method] = preds_dict
        new_data.preds = avg_preds

        # Return the updated new_data - only pred_probs and preds are changed, the rest are the same as the initially chosen new_data
        return new_data

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        self._clf = clf
        self._method = 'task_fusion'
        self._seed = seed

        self._x = data['x']
        self._y = data['y']
        self._labels = np.array(data['labels'])

        feature_names = list(self._x.columns.values)
        splitter = DataSplitterFactory().get(mode=self._mode)
        self._splits = splitter.make_splits(data=data, seed=self._seed)

        # defining metrics
        acc = []
        fms = []
        roc = []
        precision = []
        recall = []
        specificity = []

        pred = {}
        pred_prob = {}
        k_range = None

        print("Model %s" % self._clf)
        print("=========================")

        for idx, fold in enumerate(self._splits):
            print("Processing fold: %i" % idx)
            x_train, y_train = fold['x_train'], fold['y_train'].ravel()
            x_test, y_test = fold['x_test'], fold['y_test'].ravel()
            labels_train, labels_test = fold['train_labels'], fold['test_labels']

            acc_scores = []
            fms_scores = []
            roc_scores = []
            p_scores = []  # precision
            r_scores = []  # recall
            spec_scores = []

            # getting feature selected x_train, x_test and the list of selected features
            x_train_fs, x_test_fs, selected_feature_names, k_range = \
                FeatureSelector().select_features(fold_data=fold, feature_names=feature_names, k_range=k_range)

            # fit the model
            model = ClassifiersFactory.get_model(clf)
            model = model.fit(x_train_fs, y_train)

            # make predictions
            yhat = model.predict(x_test_fs)
            yhat_probs = model.predict_proba(x_test_fs)
            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat[i]
                pred_prob[labels_test[i]] = yhat_probs[i]

            # calculating metrics for each fold
            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=y_test, y_pred=yhat,
                                          y_prob=yhat_probs[:, 1], acc_saved=acc_scores,
                                          fms_saved=fms_scores, roc_saved=roc_scores,
                                          precision_saved=p_scores, recall_saved=r_scores, spec_saved=spec_scores)

            # adding every fold metric to the bigger list of metrics
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)
            precision.append(p_scores)
            recall.append(r_scores)
            specificity.append(spec_scores)

        self._save_results(method=self._method, acc=acc, fms=fms, roc=roc,
                            precision=precision, recall=recall, specificity=specificity,
                            pred=pred, pred_prob=pred_prob, k_range=k_range)

        return self

    def calculate_task_fusion_results(self, data):
        acc = []
        fms = []
        roc = []
        precision = []
        recall = []
        specificity = []

        params = ParamsHandler.load_parameters('settings')
        output_folder = params["output_folder"]
        extraction_method = params["PID_extraction_method"]
        nfolds = params['folds']

        # get list of superset_ids from the saved file
        super_pids_file_path = os.path.join('results', output_folder, extraction_method + '_super_pids.csv')
        superset_ids = list(pd.read_csv(super_pids_file_path)['interview'])

        # random shuffle based on random seed
        random.Random(self._seed).shuffle(superset_ids)
        splits = np.array_split(superset_ids, nfolds)

        method = 'task_fusion'
        pred = data.preds[method]
        pred_prob = data.pred_probs[method]
        k_range = data._best_k[method]['k_range']

        # compute performance measures for each of the splits
        for i in splits:
            acc_scores = []
            fms_scores = []
            roc_scores = []
            p_scores = []  # precision
            r_scores = []  # recall
            spec_scores = []  # specificity

            # get the prediction probabilities, predicted outcomes, and labels for each of the PIDs in this split
            y_true_sub = []
            y_pred_sub = []
            y_prob_sub = []

            for j in i:
                if j in data.y.keys():
                    y_true_sub.append(data.y[j])
                    y_pred_sub.append(data.preds[method][j])
                    y_prob_sub.append(data.pred_probs[method][j])

            y_true_sub = np.array(y_true_sub)
            y_pred_sub = np.array(y_pred_sub)
            y_prob_sub = np.array(y_prob_sub)

            # calculate the performance metrics at the fold level
            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=y_true_sub, y_pred=y_pred_sub,
                                          y_prob=y_prob_sub[:, 1], acc_saved=acc_scores,
                                          fms_saved=fms_scores, roc_saved=roc_scores,
                                          precision_saved=p_scores, recall_saved=r_scores, spec_saved=spec_scores)

            # saving performance metrics for each fold
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)
            precision.append(p_scores)
            recall.append(r_scores)
            specificity.append(spec_scores)

        # save performance metrics
        self._save_results(method, acc=acc, fms=fms, roc=roc,
                            precision=precision, recall=recall, specificity=specificity,
                            pred=pred, pred_prob=pred_prob, k_range=k_range)

        return self
