from classes.data_splitters.DataSplitter import DataSplitter
from classes.handlers.ParamsHandler import ParamsHandler

import numpy as np
import pandas as pd
import os
import random
import copy


class FusionDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, seed: int) -> list:
        self.random_seed = seed
        x = data['x']
        y = data['y']
        labels = np.array(data['labels'])
        fold_data = []

        params = ParamsHandler.load_parameters("settings")
        output_folder = params["output_folder"]
        extraction_method = params["PID_extraction_method"]
        dataset_name = params["dataset"]
        tasks = params["tasks"]

        method = 1
        splits = []

        # option 1: Superset PIDs
        if method == 1:
            # get list of superset_ids from the saved file
            # super_pids_file_path = os.path.join('assets', output_folder, extraction_method + '_super_pids.csv')
            super_pids_file_path = os.path.join(os.getcwd(), 'assets', dataset_name, 'PIDs', self.mode +
                                                '_' + extraction_method + '_super_pids.csv')
            superset_ids = list(pd.read_csv(super_pids_file_path)['interview'])

            # random shuffle based on random seed
            random.Random(self.random_seed).shuffle(superset_ids)
            splits = np.array_split(superset_ids, self.nfolds)

        # option 2: Split an intersection of pids across tasks, then split the out-of-intersection pids, then merge them equally
        if method == 2:
            pid_file_paths = {task: os.path.join('results', output_folder, extraction_method + '_' + task + '_pids.csv') for task in tasks}
            pids = [list(pd.read_csv(pid_file_paths[task])['interview']) for task in tasks]

            uni_pids = inter_pids = copy.deepcopy(pids)

            # creating intersection of pids across tasks
            while len(inter_pids) > 1:
                inter_pids = [np.intersect1d(inter_pids[i], inter_pids[i + 1]) for i in range(len(inter_pids) - 1)]
            inter_pids = list(inter_pids[0])

            # creating union of pids across tasks
            while len(uni_pids) > 1:
                uni_pids = [np.union1d(uni_pids[i], uni_pids[i+1]) for i in range(len(uni_pids) - 1)]
            uni_pids = uni_pids[0]

            # difference in uni_pids and inter_pids
            diff_pids = list(np.setxor1d(uni_pids, inter_pids))

            # shuffling before splitting
            random.Random(self.random_seed).shuffle(inter_pids)
            random.Random(self.random_seed).shuffle(diff_pids)

            inter_splits = np.array_split(inter_pids, self.nfolds)
            diff_splits = np.array_split(diff_pids, self.nfolds)

            splits = []
            for i in range(self.nfolds):
                splits.append(np.append(inter_splits[i], diff_splits[i]))

        # after creating the splits:
        # manually creating folds and filling data
        folds = [np.intersect1d(group, labels) for group in splits]
        for i in range(len(folds)):
            fold = {}

            test = folds[i]
            train = np.concatenate(folds[:i] + folds[i + 1:])

            train_index = [np.where(x.index == train[j])[0][0] for j in range(len(train))]
            test_index = [np.where(x.index == test[j])[0][0] for j in range(len(test))]

            fold['x_train'] = x.values[train_index]
            fold['y_train'] = y.values[train_index]
            fold['x_test'] = x.values[test_index]
            fold['y_test'] = y.values[test_index]
            fold['train_labels'] = labels[train_index]
            fold['test_labels'] = labels[test_index]
            fold_data.append(fold)

        return fold_data
