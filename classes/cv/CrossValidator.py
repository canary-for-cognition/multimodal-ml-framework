from classes.factories.TrainersFactory import TrainersFactory
from classes.handlers.ParamsHandler import ParamsHandler

import os
import csv
import pandas as pd
import operator
import warnings
warnings.filterwarnings("ignore")

class CrossValidator:
    def __init__(self, mode: str, classifiers: list):
        self.__trainer = None
        self.mode = mode
        self.seed = None
        self.classifiers = classifiers
        self.dataset_name = None

    def cross_validate(self, seed: int, tasks_data: dict):
        """
        :param seed: seed
        :param tasks_data: dictionary that contains data from each task
        :return: nothing
        """

        self.seed = seed
        settings = ParamsHandler.load_parameters('settings')
        self.dataset_name = settings['dataset']
        aggregation_method = settings['aggregation_method']
        meta_clf = settings['meta_classifier']
        prefixes = {'single_tasks': 'results_new_features',
                    'fusion': 'results_task_fusion',
                    'ensemble': 'results_ensemble'}
        feature_importance = False

        # running trainer for each of the tasks
        if self.mode == 'single_tasks':
            for task in tasks_data.keys():
                print("\nTask: ", task)
                print("---------------")

                task_path = os.path.join(self.dataset_name, task)
                task_params = ParamsHandler.load_parameters(task_path)
                feature_sets = task_params['features']

                # running trainer for each modality separately
                for modality, modality_data in tasks_data[task].items():
                    modality_feature_set = list(feature_sets[modality].keys())[0]

                    # training the models
                    trained_models = {}
                    for clf in self.classifiers:
                        self.__trainer = TrainersFactory().get(self.mode)
                        trained_models[clf] = self.__trainer.train(data=modality_data, clf=clf, seed=self.seed,
                                                                   feature_set=modality_feature_set,
                                                                   feature_importance=False)

                    # saving results
                    CrossValidator.save_results(self, trained_models=trained_models, feature_set=modality_feature_set,
                                                prefix=prefixes[self.mode], method='default', save_to_csv=True,
                                                get_prediction=True, feature_importance=False)

        elif self.mode == 'fusion':
            trained_models = []
            method = 'task_fusion'

            # running the trainer for each of the tasks
            for task in tasks_data.keys():
                # print("\nTask: ", task)
                # print("---------------")

                trained_models_task = []
                task_path = os.path.join(self.dataset_name, task)
                task_params = ParamsHandler.load_parameters(task_path)
                feature_sets = task_params['features']

                # running trainer for each modality separately
                for modality, modality_data in tasks_data[task].items():
                    modality_feature_set = list(feature_sets[modality].keys())[0]

                    trained_models_modality = {}
                    for clf in self.classifiers:
                        self.__trainer = TrainersFactory().get(self.mode)
                        trained_models_modality[clf] = self.__trainer.train(data=modality_data, clf=clf, seed=self.seed,
                                                                            feature_set=modality_feature_set,
                                                                            feature_importance=False)

                    # saving each modality's results
                    CrossValidator.save_results(self, trained_models=trained_models_modality,
                                                feature_set=modality_feature_set,
                                                prefix=prefixes[self.mode], method=method, save_to_csv=True,
                                                get_prediction=True, feature_importance=feature_importance)

                    trained_models_task.append(trained_models_modality)

                # aggregating modality-wise results to make task-level results
                if len(trained_models_task) > 1:
                    data = {}
                    for clf in self.classifiers:
                        data[clf] = self.__trainer.average_results(data=trained_models_task, model=clf)
                    trained_models_task = [data]

                trained_models.append(trained_models_task[0])

                # re-calculating post-averaging metrics
                trained_models_results = {}
                for clf in self.classifiers:
                    self.__trainer = TrainersFactory().get(self.mode)
                    trained_models_results[clf] = self.__trainer.calculate_task_fusion_results(data=trained_models_task[0][clf])

                CrossValidator.save_results(self, trained_models=trained_models_results, feature_set=task,
                                            prefix=prefixes[self.mode], method=method, save_to_csv=True,
                                            get_prediction=True, feature_importance=feature_importance)

            # compiling the data from all tasks here then aggregating them
            final_trained_models = {}
            for clf in self.classifiers:
                final_trained_models[clf] = self.__trainer.average_results(data=trained_models, model=clf)

            # recalculating metrics and results after aggregation
            final_trained_models_results = {}
            for clf in self.classifiers:
                self.__trainer = TrainersFactory().get(self.mode)
                final_trained_models_results[clf] = self.__trainer.calculate_task_fusion_results(data=final_trained_models[clf])

            # saving results after full aggregation
            CrossValidator.save_results(self, trained_models=final_trained_models_results, feature_set='',
                                        prefix=prefixes[self.mode], method=method, save_to_csv=True,
                                        get_prediction=True, feature_importance=feature_importance)

        elif self.mode == 'ensemble':
            """
            Work in progress
            """
            final_stacked = {}
            trained_models = []

            for task in tasks_data.keys():
                # print("\nTask: ", task)
                # print("---------------")

                task_stacked = {}
                trained_models_task = []
                task_path = os.path.join(self.dataset_name, task)
                task_params = ParamsHandler.load_parameters(task_path)
                feature_sets = task_params['features']

                # modality_stacked = {}
                for modality, modality_data in tasks_data[task].items():
                    modality_stacked = {}
                    modality_feature_set = list(feature_sets[modality].keys())[0]

                    trained_models_modality = {}
                    for clf in self.classifiers:
                        trainer = TrainersFactory().get(self.mode)
                        trained_models_modality[clf] = trainer.train(data=modality_data, clf=clf,
                                                                     feature_set=modality_feature_set,
                                                                     feature_importance=False, seed=self.seed)

                    modality_meta_trainer = TrainersFactory().get(aggregation_method)
                    modality_stacked[modality_feature_set] = modality_meta_trainer.train(data=trained_models_modality,
                                                                                         clf=meta_clf,
                                                                                         seed=self.seed,
                                                                                         feature_set=modality_feature_set,
                                                                                         feature_importance=False)

                    CrossValidator.save_results(self, trained_models=modality_stacked,
                                                feature_set=modality_feature_set,
                                                prefix=prefixes[self.mode], method=self.mode, save_to_csv=True,
                                                get_prediction=True, feature_importance=feature_importance)

                    trained_models_task.append(modality_stacked)

                if len(trained_models_task) > 1:
                    mod_stacked_dict = {}
                    for data in trained_models_task:
                        mod_stacked_dict.update(data)

                    task_meta_trainer = TrainersFactory().get(aggregation_method)
                    task_stacked[task] = task_meta_trainer.train(data=mod_stacked_dict, clf=meta_clf,
                                                                 seed=self.seed)

                else:
                    task_stacked[task] = list(trained_models_task[0].values())[0]

                CrossValidator.save_results(self, trained_models=task_stacked, feature_set=task,
                                            prefix=prefixes[self.mode], method=self.mode, save_to_csv=True,
                                            get_prediction=True, feature_importance=feature_importance)

                trained_models.append(task_stacked)

            if len(trained_models) > 1:
                task_stacked_dict = {}
                for data in trained_models:
                    task_stacked_dict.update(data)
                final_meta_trainer = TrainersFactory().get(aggregation_method)
                final_stacked[aggregation_method] = final_meta_trainer.train(data=task_stacked_dict, clf=meta_clf,
                                                                             seed=self.seed)

            # choose what feature set to use here, idk
            CrossValidator.save_results(self, trained_models=final_stacked, feature_set='',
                                        prefix=prefixes[self.mode], method=self.mode, save_to_csv=True,
                                        get_prediction=True, feature_importance=feature_importance)

    def save_results(self, trained_models, feature_set, prefix, method='default',
                     save_to_csv=False, get_prediction=False, feature_importance=False):
        """
        :param trained_models: a dictionary of Trainer objects that are already trained
        :param feature_set: the set of features used for the specific modality/task
        :param prefix: prefix to use for saving the results into file
        :param method: variable used for referring to keys in the dict
        :param save_to_csv: bool that decides if the results are to be saved to csv or not
        :param get_prediction: bool that decides if predictions are to be saved or not
        :param feature_importance: bool that decides if feature importance values are to be saved or not
        :return: nothing
        """

        # required values
        feat_csv_writer = None
        feat_f = None
        feat_fold_csv_writer = None
        pred_csv_writer = None
        pred_f = None
        feat_fold_f = None

        params = ParamsHandler.load_parameters('settings')
        output_folder = params['output_folder']

        prediction_prefix = 'predictions'
        feature_fold_prefix = 'features_fold'
        feature_prefix = 'features'
        metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity']
        results_path = os.path.join(os.getcwd(), 'results', self.dataset_name, output_folder, str(self.seed))

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        dfs = []
        name = "%s_%s" % (prefix, feature_set)
        pred_f_file = os.path.join(results_path, prediction_prefix + "_" + name + ".csv")
        feat_fold_f_file = os.path.join(results_path, feature_fold_prefix + "_" + name + ".csv")

        if get_prediction:
            pred_f = open(pred_f_file, 'w')
            pred_csv_writer = csv.writer(pred_f)
            headers = ['model', 'PID', 'prob_0', 'prob_1', 'pred']
            pred_csv_writer.writerow(headers)

        if feature_importance:
            feat_fold_f = open(feat_fold_f_file, 'w')
            feat_fold_csv_writer = csv.writer(feat_fold_f)
            headers_fold = ['model', 'fold', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'p_value']
            feat_fold_csv_writer.writerow(headers_fold)

            feat_f = open('%s_%s.csv' % (os.path.join(results_path, feature_prefix), name), 'w')
            feat_csv_writer = csv.writer(feat_f)
            headers = ['model', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'p_value']
            feat_csv_writer.writerow(headers)

        for model in trained_models:
            cv = trained_models[model]
            # k_range = cv.best_k[method]['k_range']
            k_range = [1]
            for metric in metrics:
                if metric in cv.results[method].keys():
                    results = cv.results[method][metric]
                    # print('@@', metric, results)
                    df = pd.DataFrame(results, columns=k_range)
                    df['metric'] = metric
                    df['model'] = model
                    dfs += [df]

            if get_prediction:
                for pid, prob in cv.pred_probs[method].items():
                    prob_0 = prob[0]
                    prob_1 = prob[1]
                    pred = cv.preds[method][pid]
                    row = [model, pid, prob_0, prob_1, pred]
                    pred_csv_writer.writerow(row)

            if feature_importance and cv.feature_scores_fold[method] is not None and cv.feature_scores_all[method] is not None:
                i = 0
                for feat_score in cv.feature_scores_fold[method]:
                    sorted_feat_score = sorted(feat_score.items(),
                                               key=operator.itemgetter(1), reverse=True)
                    for feat, score in sorted_feat_score:
                        row = [model, i, feat, score[0], score[1], score[2], score[3], score[4], score[5]]
                        feat_fold_csv_writer.writerow(row)
                    i += 1
                sorted_feat_score_all = \
                    sorted(cv.feature_scores_all[method].items(),
                           key=operator.itemgetter(1),
                           reverse=True)
                for feat, score in sorted_feat_score_all:
                    row = [model, feat, score[0], score[1], score[2], score[3], score[4], score[5]]
                    feat_csv_writer.writerow(row)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        if save_to_csv:
            df.to_csv(os.path.join(results_path, name + '.csv'), index=False)
        if get_prediction:
            pred_f.close()
        if feature_importance:
            feat_f.close()
            feat_fold_f.close()

