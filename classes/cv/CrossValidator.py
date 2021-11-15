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
        self.__mode, self.__classifiers = mode, classifiers
        supp_modes = ["single_task", "fusion", "ensemble"]
        if self.__mode not in supp_modes:
            raise ValueError("Mode '{}' is not supported! Supported modes are '{}'".format(self.__mode, supp_modes))

        self.__seed, self.__dataset_name, self.__feature_importance, self.__path_to_results = None, None, None, None
        self.__trainer = TrainersFactory().get(self.__mode)
        self.__prefixes = dict(single_tasks='results_new_features',
                               fusion='results_task_fusion',
                               ensemble='results_ensemble')
        self.__metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity']
        self.__headers = {
            "pred": ['model', 'PID', 'prob_0', 'prob_1', 'pred'],
            "feat": ['model', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'p_value'],
            "feat_fold": ['model', 'fold', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'pvalue']
        }


    def __single_task(self, tasks_data: dict):
        for task in tasks_data.keys():
            print("\n *** Task: {} ***".format(task))
            print("------------------------------------------------------------\n")

            feature_sets = ParamsHandler.load_parameters(os.path.join(self.__dataset_name, task))['features']

            # Running trainers for each modality separately
            for modality, modality_data in tasks_data[task].items():
                modality_feature_set = list(feature_sets[modality].keys())[0]

                # Training the models
                trained_models = {}
                for clf in self.__classifiers:
                    trained_models[clf] = self.__trainer.train(modality_data, clf, self.__seed,
                                                               feature_set=modality_feature_set,
                                                               feature_importance=False)

                CrossValidator.__save_results(self, trained_models, modality_feature_set, feat_imp=False)

    def __train_modalities(self, tasks_data: dict, feature_sets: dict) -> list:
        trained_models_task = []
        for modality, modality_data in tasks_data.items():
            modality_feature_set = list(feature_sets[modality].keys())[0]

            trained_models_modality = {}
            for clf in self.__classifiers:
                trained_models_modality[clf] = self.__trainer.train(modality_data, clf, self.__seed,
                                                                    feature_set=modality_feature_set,
                                                                    feature_importance=False)

            # Saving each modality's results
            self.__save_results(trained_models_modality, modality_feature_set, method='task_fusion')

            trained_models_task.append(trained_models_modality)
        return trained_models_task

    def __fusion(self, tasks_data: dict):
        trained_models = []

        for task in tasks_data.keys():
            feature_sets = ParamsHandler.load_parameters(os.path.join(self.__dataset_name, task))['features']

            # Running trainers for each modality separately
            trained_models_task = self.__train_modalities(tasks_data[task], feature_sets)

            # Aggregating modality-wise results to make task-level results
            if len(trained_models_task) > 1:
                data = {}
                for clf in self.__classifiers:
                    data[clf] = self.__trainer.average_results(trained_models_task, clf)
                trained_models_task = [data]

            trained_models.append(trained_models_task[0])

            # Re-calculating post-averaging metrics
            trained_models_results = {}
            for clf in self.__classifiers:
                trained_models_results[clf] = self.__trainer.calculate_task_fusion_results(trained_models_task[0][clf])

            self.__save_results(trained_models_results, task, method='task_fusion')

            # Compiling the data from all tasks here then aggregating them
            final_trained_models = {}
            for clf in self.__classifiers:
                final_trained_models[clf] = self.__trainer.average_results(trained_models, clf)

            # Recalculating metrics and results after aggregation
            final_trained_models_results = {}
            for clf in self.__classifiers:
                final_trained_models_results[clf] = self.__trainer.calculate_task_fusion_results(
                    final_trained_models[clf])

            # Saving results after full aggregation
            self.__save_results(final_trained_models_results, feature_set='', method='task_fusion')

    def __ensemble(self, tasks_data: dict, aggregation_method: str, meta_clf: str):
        final_stacked, trained_models = {}, []

        for task in tasks_data.keys():
            task_stacked, trained_models_task = {}, []

            feature_sets = ParamsHandler.load_parameters(os.path.join(self.__dataset_name, task))['features']

            for modality, modality_data in tasks_data[task].items():
                modality_stacked = {}
                modality_feature_set = list(feature_sets[modality].keys())[0]

                trained_models_modality = {}
                for clf in self.__classifiers:
                    trainer = TrainersFactory().get(self.__mode)
                    trained_models_modality[clf] = trainer.train(data=modality_data, clf=clf,
                                                                 feature_set=modality_feature_set,
                                                                 feature_importance=False, seed=self.__seed)

                modality_meta_trainer = TrainersFactory().get(aggregation_method)
                modality_stacked[modality_feature_set] = modality_meta_trainer.train(data=trained_models_modality,
                                                                                     clf=meta_clf,
                                                                                     seed=self.__seed,
                                                                                     feature_set=modality_feature_set,
                                                                                     feature_importance=False)

                self.__save_results(modality_stacked, modality_feature_set, method="ensemble")
                trained_models_task.append(modality_stacked)

            if len(trained_models_task) > 1:
                mod_stacked_dict = {}
                for data in trained_models_task:
                    mod_stacked_dict.update(data)

                task_meta_trainer = TrainersFactory().get(aggregation_method)
                task_stacked[task] = task_meta_trainer.train(mod_stacked_dict, meta_clf, self.__seed)
            else:
                task_stacked[task] = list(trained_models_task[0].values())[0]

            self.__save_results(task_stacked, task, method="ensemble")
            trained_models.append(task_stacked)

        if len(trained_models) > 1:
            task_stacked_dict = {}
            for data in trained_models:
                task_stacked_dict.update(data)
            final_meta_trainer = TrainersFactory().get(aggregation_method)
            final_stacked[aggregation_method] = final_meta_trainer.train(task_stacked_dict, meta_clf, self.__seed)

        self.__save_results(final_stacked, feature_set='', method="ensemble")

    def cross_validate(self, seed: int, tasks_data: dict, feature_importance: bool = False):
        settings = ParamsHandler.load_parameters('settings')
        self.__seed, self.__dataset_name, self.__feature_importance = seed, settings['dataset'], feature_importance

        output_folder = ParamsHandler.load_parameters("settings")["output_folder"]
        self.__path_to_results = os.path.join("results", self.__dataset_name, output_folder, str(self.__seed))

        if not os.path.exists(self.__path_to_results):
            os.makedirs(self.__path_to_results)

        if self.__mode == 'single_tasks':
            self.__single_task(tasks_data)
        elif self.__mode == 'fusion':
            self.__fusion(tasks_data)
        elif self.__mode == 'ensemble':
            aggregation_method = settings['aggregation_method']
            meta_clf = settings['meta_classifier']
            self.__ensemble(tasks_data, aggregation_method, meta_clf)

    def __save_results(self, trained_models, feature_set, method='default', feat_imp=None):
        """
        :param trained_models: a dictionary of Trainer objects that are already trained
        :param feature_set: the set of features used for the specific modality/task
        :param method: variable used for referring to keys in the dict
        :param feat_imp: bool that decides if feature importance values are to be saved or not
        :return: nothing
        """
        if feat_imp is None:
            feat_imp = self.__feature_importance

        name = "{}_{}".format(self.__prefixes[self.__mode], feature_set)
        pred_csv_writer = csv.writer(open(os.path.join(self.__path_to_results, "predictions_{}.csv".format(name)), 'w'))
        pred_csv_writer.writerow(self.__headers["pred"])

        feat_fold_csv_writer, feat_csv_writer = None, None
        if feat_imp:
            feat_fold_csv_writer = csv.writer(
                open(os.path.join(self.__path_to_results, "features_fold_{}.csv".format(name)), 'w'))
            feat_fold_csv_writer.writerow(self.__headers["feat_fold"])

            feat_csv_writer = csv.writer(
                open('{}_{}.csv'.format(os.path.join(self.__path_to_results, "features"), name), 'w'))
            feat_csv_writer.writerow(self.__headers["feat_fold"])

        dfs = []
        for model in trained_models:
            cv = trained_models[model]

            k_range = [1]
            for metric in self.__metrics:
                if metric in cv.results[method].keys():
                    df = pd.DataFrame(cv.results[method][metric], columns=k_range)
                    df['metric'], df['model'] = metric, model
                    dfs += [df]

            for pid, prob in cv.pred_probs[method].items():
                row = [model, pid, prob[0], prob[1], cv.preds[method][pid]]
                pred_csv_writer.writerow(row)

            if feat_imp and cv.feature_scores_fold[method] is not None and cv.feature_scores_all[method] is not None:
                for i, feat_score in enumerate(cv.feature_scores_fold[method]):
                    for feat, score in sorted(feat_score.items(), key=operator.itemgetter(1), reverse=True):
                        row = [model, i, feat, score[0], score[1], score[2], score[3], score[4], score[5]]
                        feat_fold_csv_writer.writerow(row)

                for f, score in sorted(cv.feature_scores_all[method].items(), key=operator.itemgetter(1), reverse=True):
                    row = [model, f, score[0], score[1], score[2], score[3], score[4], score[5]]
                    feat_csv_writer.writerow(row)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df.to_csv(os.path.join(self.__path_to_results, '{0}.csv'.format(name)), index=False)
