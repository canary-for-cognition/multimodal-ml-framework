import warnings

from tqdm import tqdm

from classes.factories.DataSplitterFactory import DataSplitterFactory
from classes.factories.ClassifiersFactory import ClassifiersFactory
from classes.handlers.ParamsHandler import ParamsHandler
from classes.trainers.Trainer import Trainer

warnings.filterwarnings("ignore")


class StackingTrainer(Trainer):
    def __init__(self):
        super().__init__()
        params = ParamsHandler.load_parameters('settings')
        self.__aggregation_method = params['aggregation_method']

    def train(self, data: dict, clf: str, seed: int, feature_set: str = '', feature_importance: bool = True):
        self._method = 'ensemble'

        clfs = list(data.keys())
        some_clf = clfs[0]

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

        k_range = list(data[some_clf]._best_k.values())[0]['k_range']

        splitter = DataSplitterFactory().get(self.__aggregation_method)
        self._splits = splitter.make_splits(data=data, seed=self._seed)

        for idx, fold in enumerate(tqdm(self._splits, desc='Stacking Trainer')):
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

            # train
            meta_model = ClassifiersFactory().get_model(clf)
            meta_model = meta_model.fit(x_train, y_train)

            # make predictions
            yhat_preds = meta_model.predict(x_test)
            yhat_pred_probs = meta_model.predict_proba(x_test)

            for i in range(labels_test.shape[0]):
                pred[labels_test[i]] = yhat_preds[i]
                pred_prob[labels_test[i]] = yhat_pred_probs[i]

            # calculating metrics for each fold

            acc_scores, fms_scores, roc_scores, p_scores, r_scores, spec_scores = \
                self.compute_save_results(y_true=y_test, y_pred=yhat_preds,
                                          y_prob=yhat_pred_probs[:, 1], acc_saved=acc_scores,
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
