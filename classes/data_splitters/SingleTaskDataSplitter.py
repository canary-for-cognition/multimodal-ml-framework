from classes.data_splitters.DataSplitter import DataSplitter

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold


class SingleTaskDataSplitter(DataSplitter):
    def __init__(self):
        super().__init__()

    def make_splits(self, data: dict, seed: int) -> list:
        self.random_seed = seed
        x = data['x']
        y = data['y']
        labels = np.array(data['labels'])
        fold_data = []

        folds = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=self.random_seed).split(x, y, groups=labels)
        output_file = os.path.join(os.getcwd(), 'assets', self.mode + 'cv_info.txt')

        with open(output_file, 'w') as f:
            cnt = 0
            for train_index, test_index in folds:
                fold = {
                    'x_train': x.values[train_index],
                    'y_train': y.values[train_index],
                    'x_test': x.values[test_index],
                    'y_test': y.values[test_index],
                    'train_labels': labels[train_index],
                    'test_labels': labels[test_index]
                }
                fold_data.append(fold)
                f.write('Train%d, %s\n' % (cnt, x.index[train_index].tolist()))
                f.write('Test%d, %s\n' % (cnt, x.index[test_index].tolist()))
                cnt += 1

        return fold_data
