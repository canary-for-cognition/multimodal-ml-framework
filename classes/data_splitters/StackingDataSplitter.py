import copy
import os
import random

import numpy as np
import pandas as pd

from classes.data_splitters.DataSplitter import DataSplitter
from classes.handlers.ParamsHandler import ParamsHandler


class StackingDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, seed: int) -> list:
        self.random_seed = seed
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
            super_pids_file_path = os.path.join(os.getcwd(), 'assets', dataset_name, 'PIDs', self._mode +
                                                '_' + extraction_method + '_super_pids.csv')
            superset_ids = list(pd.read_csv(super_pids_file_path)['interview'])

            # random shuffle based on random seed
            random.Random(self.random_seed).shuffle(superset_ids)
            splits = np.array_split(superset_ids, self._num_folds)

        # create and fill x and y
        # x: from superset ids, find the pids in each of the given trained models and get their prediction values,
        # y: from superset ids, find the pids in each of the given trained models and get their y values
        # labels: just superset_ids
        data_path = os.path.join('datasets', dataset_name)
        diag = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))
        true_y = diag['diagnosis'] != 'HC'
        true_y.index = diag['interview']

        x = np.empty(shape=(len(superset_ids), len(data) + 1), dtype='object')
        x[:] = np.NaN
        y = np.empty(shape=(len(superset_ids), 2), dtype='object')
        labels = np.empty(shape=len(superset_ids), dtype='object')

        for p in range(len(superset_ids)):
            x[p][0] = y[p][0] = labels[p] = superset_ids[p]
            for k, tr in enumerate(data.values()):
                pids = list(tr.preds['ensemble'].keys())
                if superset_ids[p] in pids:
                    x[p][k + 1] = tr.preds['ensemble'][superset_ids[p]]
                    y[p][1] = true_y[superset_ids[p]]

        # now, for some modalities where the number of PIDs are less than the union (162), the 'within modality' x will
        # have rows where all columns will be None
        # that just verifies that those PIDs don't exist for that modality
        # so if there's rows with all None values, skip those PIDs

        # but in the case of 'within task', there may be some PIDs that are missing in one modality but not in the other
        # in that case, not all columns will have None in them
        # if that happens, either: (a) fill in random values, or (b) choose by voting

        to_remove_all = [r for r in range(len(x)) if np.all(np.isnan(x[r, 1:].astype('float')))]
        to_fill_any = [r for r in range(len(x)) if np.any(np.isnan(x[r, 1:].astype('float')))]

        # within modality level, when all classifiers don't have a certain PID. That PID gets counted in to_remove_all
        if len(to_remove_all) > 0:
            x = np.delete(x, to_remove_all, axis=0)
            y = np.delete(y, to_remove_all, axis=0)
            labels = np.delete(labels, to_remove_all, axis=0)

        # within task or across task level, when some (not all) data keys don't have a certain PID.
        if len(to_fill_any) > 0:
            indices_of_nans = np.argwhere(np.isnan(x[:, 1:].astype(float)))
            for row, col in indices_of_nans:
                # if (a), fill in random values
                # x[row, col+1] = bool(random.getrandbits(1))

                # if (b), choose the the most commonly found value (by voting)
                x_row_as_float = x[row, 1:].astype(float)
                nan_mask_row = np.invert(np.isnan(x_row_as_float))
                x_masked_row = x_row_as_float[nan_mask_row].astype(int)
                chosen_val = np.argmax(np.bincount(x_masked_row))
                x[row, col + 1] = bool(chosen_val)

        x = pd.DataFrame(x[:, 1:], index=x[:, 0], dtype='bool')
        y = pd.Series(y[:, 1], index=y[:, 0], dtype='bool')

        # option 2: Split an intersection of pids across tasks, then split the out-of-intersection pids,
        # then merge them equally
        if method == 2:
            pid_file_paths = {task: os.path.join('results', output_folder, extraction_method +
                                                 '_' + task + '_pids.csv') for task in tasks}
            pids = [list(pd.read_csv(pid_file_paths[task])['interview']) for task in tasks]

            uni_pids = inter_pids = copy.deepcopy(pids)

            # creating intersection of pids across tasks
            while len(inter_pids) > 1:
                inter_pids = [np.intersect1d(inter_pids[i], inter_pids[i + 1]) for i in range(len(inter_pids) - 1)]
            inter_pids = list(inter_pids[0])

            # creating union of pids across tasks
            while len(uni_pids) > 1:
                uni_pids = [np.union1d(uni_pids[i], uni_pids[i + 1]) for i in range(len(uni_pids) - 1)]
            uni_pids = uni_pids[0]

            # difference in uni_pids and inter_pids
            diff_pids = list(np.setxor1d(uni_pids, inter_pids))

            # shuffling before splitting
            random.Random(self.random_seed).shuffle(inter_pids)
            random.Random(self.random_seed).shuffle(diff_pids)

            inter_splits = np.array_split(inter_pids, self._num_folds)
            diff_splits = np.array_split(diff_pids, self._num_folds)

            splits = []
            for i in range(self._num_folds):
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
