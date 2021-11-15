import warnings

import numpy as np
from tqdm import tqdm

from classes.cv.FeatureSelector import FeatureSelector
from classes.factories.ClassifiersFactory import ClassifiersFactory
from classes.factories.DataSplitterFactory import DataSplitterFactory
from classes.trainers.Trainer import Trainer

warnings.filterwarnings("ignore")


class ModelEnsembleTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self._fold_preds_train, self._fold_pred_probs_train = [], []
        self._fold_preds_test, self._fold_pred_probs_test = [], []
        self._x_train_fs, self._x_test_fs, self._y_train, self._y_test = [], [], [], []

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        self._method = 'ensemble'
        self._seed = seed
        self._clf = clf

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
        feature_scores_fold = []
        k_range = None

        for idx, fold in enumerate(tqdm(self._splits, desc=self._clf)):
            # print("Processing fold: %i" % idx)
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

            # saving after-feature selection important values
            self._x_train_fs.append(x_train_fs)
            self._y_train.append(y_train)
            self._x_test_fs.append(x_test_fs)
            self._y_test.append(y_test)

            # fit the model
            model = ClassifiersFactory().get_model(clf)
            model = model.fit(x_train_fs, y_train)
            self._models.append(model)

            # make predictions
            yhat = model.predict(x_test_fs)
            yhat_probs = model.predict_proba(x_test_fs)

            # make training predictions
            yhat_train = model.predict(x_train_fs)
            yhat_train_probs = model.predict_proba(x_train_fs)

            # for stacking
            pred_train = {}
            pred_prob_train = {}
            pred_test = {}
            pred_prob_test = {}

            # predictions train data for stacking
            for i in range(labels_train.shape[0]):
                pred_train[labels_train[i]] = yhat_train[i]
                pred_prob_train[labels_train[i]] = yhat_train_probs[i]

            self._fold_preds_train.append(pred_train)
            self._fold_pred_probs_train.append(pred_prob_train)

            # predictions test data for stacking, and normal
            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat[i]
                pred_prob[labels_test[i]] = yhat_probs[i]

                pred_test[labels_test[i]] = yhat[i]
                pred_prob_train[labels_test[i]] = yhat_probs[i]

            self._fold_preds_test.append(pred_test)
            self._fold_pred_probs_test.append(pred_prob_test)

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
            #     feature_scores_fold.append(self.save_feature_importance(x=x_train_fs, y=None, clf=model,
            #                                                             feature_names=selected_feature_names))
            '''

        self._save_results(method=self._method, acc=acc, fms=fms, roc=roc,
                           precision=precision, recall=recall, specificity=specificity,
                           pred=pred, pred_prob=pred_prob, k_range=k_range)

        self._feature_scores_fold[self._method] = feature_scores_fold

        '''
        # if feature_importance:  # get feature importance from the whole data
        #     self._feature_scores_all[self._method] = \
        #         self.save_feature_importance(x=self._x, y=self._y,
        #                                      clf=clf, feature_names=feature_names)
        '''

        return self
