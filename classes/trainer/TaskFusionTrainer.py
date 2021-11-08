from classes.trainer.Trainer import Trainer
from classes.cv.FeatureSelector import FeatureSelector
from classes.handlers.ModelsHandler import ModelsHandler
from classes.handlers.ParamsHandler import ParamsHandler
from classes.factories.DataSplitterFactory import DataSplitterFactory

import numpy as np
import random
import os
import pandas as pd


class TaskFusionTrainer(Trainer):
    def __init__(self):
        super().__init__()

    '''
    # def save_feature_importance(self, x, y, model_name, model, feature_names):
    #     if model is None:
    #         X_fs, feature_names = self.do_feature_selection_all(x.values, y,
    #                                                             feature_names)
    #         model = get_classifier(model_name)
    #         model.fit(X_fs, y)
    #         X = X_fs
    #     feature_scores = get_feature_scores(model_name, model, feature_names, x)
    #     return feature_scores
    '''

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        self.clf = clf
        self.method = 'task_fusion'
        self.seed = seed

        self.x = data['x']
        self.y = data['y']
        self.labels = np.array(data['labels'])

        feature_names = list(self.x.columns.values)
        splitter = DataSplitterFactory().get(mode=self.mode)
        self.splits = splitter.make_splits(data=data, seed=self.seed)

        # defining metrics
        acc = []
        fms = []
        roc = []
        precision = []
        recall = []
        specificity = []

        pred = {}
        pred_prob = {}
        # feature_scores_fold = []
        k_range = None

        print("Model %s" % self.clf)
        print("=========================")

        for idx, fold in enumerate(self.splits):
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
            model = ModelsHandler.get_model(clf)
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

            '''
            # if feature_importance:
            #     feature_scores_fold.append(self.save_feature_importance(X=X_train_fs,
            #                                                             y=None, model_name=model, model=clf,
            #                                                             feature_names=selected_feature_names))
            '''

        self.save_results(method=self.method, acc=acc, fms=fms, roc=roc,
                          precision=precision, recall=recall, specificity=specificity,
                          pred=pred, pred_prob=pred_prob, k_range=k_range)

        '''
        # self.feature_scores_fold[self.method] = feature_scores_fold

        # if feature_importance:  # get feature importance from the whole data
        #     self.feature_scores_all[self.method] = \
        #         self.save_feature_importance(X=self.X, y=self.y,
        #                                      model_name=model, model=None, feature_names=feature_names)
        '''

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
        random.Random(self.seed).shuffle(superset_ids)
        splits = np.array_split(superset_ids, nfolds)

        method = 'task_fusion'
        pred = data.preds[method]
        pred_prob = data.pred_probs[method]
        k_range = data.best_k[method]['k_range']

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
        self.save_results(method, acc=acc, fms=fms, roc=roc,
                          precision=precision, recall=recall, specificity=specificity,
                          pred=pred, pred_prob=pred_prob, k_range=k_range)

        return self