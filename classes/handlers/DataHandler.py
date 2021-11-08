from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.PIDExtractor import PIDExtractor

from typing import List
import pandas as pd
import os


class DataHandler:

    def __init__(self, mode: str, output_folder: str, extraction_method: str):
        self.mode = mode
        self.output_folder = output_folder
        self.extraction_method = extraction_method
        self.pid_file_paths = None
        self.dataset_name = ParamsHandler.load_parameters('settings')['dataset']

    def load_data(self, tasks: List) -> dict:
        tasks_data = {task: None for task in tasks}
        self.pid_file_paths = {task: os.path.join('assets', self.dataset_name, 'PIDs', self.mode + '_' +
                                                  self.extraction_method + '_' + task + '_pids.csv') for task in tasks}

        # extract PIDs
        PIDExtractor(mode=self.mode, extraction_method=self.extraction_method, output_folder=self.output_folder,
                     pid_file_paths=self.pid_file_paths, dataset_name=self.dataset_name).get_list_of_pids(tasks=tasks)

        for task in tasks:
            print(task)
            task_path = os.path.join(self.dataset_name, task)
            params = ParamsHandler.load_parameters(task_path)
            modalities = params['modalities']
            feature_set = params['feature_sets']

            modality_data = {modality: None for modality in modalities}
            for modality in modalities:
                modality_data[modality] = self.get_data(task, modality, feature_set, self.pid_file_paths[task])
                tasks_data[task] = modality_data

        return tasks_data


    @staticmethod
    def get_data(task: str, modality: str, feature_set: dict, pid_file_path: str) -> dict:

        dataset_name = ParamsHandler.load_parameters('settings')['dataset']
        feature_path = os.path.join(dataset_name, 'feature_sets')
        feature_subsets_path = os.path.join(feature_path, 'feature_subsets')
        
        data_path = os.path.join('datasets', dataset_name)

        # get pids from a saved file, which was created by get_list_of_pids based on the conditions given to it
        pids = pd.read_csv(pid_file_path)

        # initializing the dataset as the list of PIDs
        dataset = pids
        final_features = []
        features = list(feature_set.values())

        # unpacking all features from their feature sets into final_features
        for feat in features:
            features_subset = ParamsHandler.load_parameters(os.path.join(feature_path, feat))
            final_features.extend(features_subset)

        if modality == 'eye':
            for feat in final_features:
                to_select = ['interview']
                if feat.startswith('eye'):
                    print("--", feat)
                    to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))
                    eye_data = pd.read_csv(os.path.join(data_path, feat + '.csv'))
                    eye_dataset = eye_data.loc[eye_data['task'] == task]

                    eye_dataset = eye_dataset[to_select]
                    dataset = pd.merge(dataset, eye_dataset, on='interview')

        elif modality == 'speech':
            
            # NLP data files merging. No need to put it in the loop as that adds time
            text_data = pd.read_csv(os.path.join(data_path, 'text.csv'))
            acoustic_data = pd.read_csv(os.path.join(data_path, 'acoustic.csv'))
            
            if dataset_name == 'canary':
                task_mod_dict = {'CookieTheft': 1, 'Reading': 2, 'Memory': 3}
                task_mod = task_mod_dict[task]

                lang_merged = pd.merge(text_data, acoustic_data, on=['interview', 'task'])

            elif dataset_name == 'dementia_bank':
                discourse_data = pd.read_csv(os.path.join(data_path, 'discourse.csv'))
                demographic_data = pd.read_csv(os.path.join(data_path, 'demographic.csv'))

                lang_merged = pd.merge(text_data, acoustic_data, on=['interview'])
                lang_merged = pd.merge(lang_merged, discourse_data, on=['interview'])
                lang_merged = pd.merge(lang_merged, demographic_data, on=['interview'])

            for feat in final_features:
                to_select = ['interview']
                if not feat.startswith('eye'):
                    print("--", feat)
                    to_select.extend(ParamsHandler.load_parameters(os.path.join(feature_subsets_path, feat)))

                    if feat == 'fraser':
                        fraser_data = pd.read_csv(os.path.join(data_path, feat + '.csv'))
                        fraser_dataset = fraser_data.loc[fraser_data['task'] == task]

                        fraser_dataset = fraser_dataset[to_select]
                        dataset = pd.merge(dataset, fraser_dataset, on='interview')
                        continue

                    if dataset_name == 'canary':
                        lang_merged = lang_merged.loc[lang_merged['task'] == task_mod]
                    lang_dataset = lang_merged[to_select]
                    dataset = pd.merge(dataset, lang_dataset, on='interview')

        # merging with diagnosis to get patients with valid diagnosis
        diagnosis_data = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))
        dataset = pd.merge(dataset, diagnosis_data, on='interview')

        # random sample
        dataset = dataset.sample(frac=1, random_state=10)
        
        if dataset_name == 'canary':
            # labels
            labels = list(dataset['interview'])

            # y
            y = dataset['diagnosis'] != 'HC'

            # x
            drop = ['interview', 'diagnosis', 'Unnamed: 0_x', 'Unnamed: 0_y']

        elif dataset_name == 'dementia_bank':
            # labels
            labels = [label[:3] for label in dataset['interview']]

            # y
            y = dataset['diagnosis'] != 'Control'

            # x
            drop = ['interview', 'diagnosis', 'Unnamed: 0', 'gender', 'gender_int']

        x = dataset.drop(drop, axis=1, errors='ignore')
        x = x.apply(pd.to_numeric, errors='ignore')

        x.index = labels
        y.index = labels

        return {'x': x, 'y': y, 'labels': labels}
