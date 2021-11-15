import numpy as np

from classes.cv.FeatureSelector import FeatureSelector
from classes.factories.ClassifiersFactory import ClassifiersFactory
from classes.factories.DataSplitterFactory import DataSplitterFactory
from classes.trainers.Trainer import Trainer


class SingleModelTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        self._clf, self._method, self._seed = clf, 'default', seed

        self._x, self._y = data['x'], = data['y']
        self._labels = np.array(data['labels'])

        feature_names = list(self._x.columns.values)
        splitter = DataSplitterFactory().get(mode=self._mode)
        self._splits = splitter.make_splits(data=data, seed=self._seed)

        acc, fms, roc, precision, recall, specificity = [], [], [], [], [], []
        pred, pred_prob = {}, {}
        feature_scores_fold = []
        k_range = None

        print("Model {}".format(self._clf))
        print("==================================================")

        for idx, fold in enumerate(self._splits):
            print("Processing fold: {:d}".format(idx))
            x_train, y_train = fold['x_train'], fold['y_train'].ravel()
            x_test, y_test = fold['x_test'], fold['y_test'].ravel()
            labels_train, labels_test = fold['train_labels'], fold['test_labels']

            acc_scores, fms_scores, roc_scores, r_scores, spec_scores = [], [], [], [], []

            # getting feature selected x_train, x_test and the list of selected features
            x_train_fs, x_test_fs, selected_feature_names, k_range = \
                FeatureSelector().select_features(fold_data=fold, feature_names=feature_names, k_range=k_range)

            # fit the model
            model = ClassifiersFactory.get_model(clf)
            model = model.fit(x_train_fs, y_train)
            self._models.append(model)

            # make predictions
            yhat = model.predict(x_test_fs)
            yhat_probs = model.predict_proba(x_test_fs)
            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat[i]
                pred_prob[labels_test[i]] = yhat_probs[i]

            self.preds.append(pred)
            self.pred_probs.append(pred_prob)

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

        self._feature_scores_fold[self._method] = feature_scores_fold

        return self
