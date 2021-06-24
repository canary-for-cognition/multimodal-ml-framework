# noinspection PyUnusedLocal
from typing import List

import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler


class DataHandler:

    def __init__(self):
        pass

    def load_data(self, tasks: List, mode: str) -> List:
        for task in tasks:
            params = ParamsHandler.load_parameters(task)
            modalities = params['modalities']
            features = params['features']
            feature_set = params['feature_sets']

            pids = self.get_list_of_pids(mode, modalities, task)
            data = self.get_data(task, feature_set, pids)

        if mode == "single_tasks":
            pass
        if mode == "fusion":
            pass
        if mode == "ensemble":
            pass
        return []

    @staticmethod
    def get_list_of_pids(mode: str, modalities: dict, task: str) -> List:
        """
        :param mode: mode specifies how the PIDs should be handled (single = intersect everything, fusion = union of modality level PIDs, intersect within modality)
        :param modalities: which modalities (based on the task) would influence selected PIDs
        :param task: the task for which PIDs are required
        :return: list of PIDs that satisfy the task and modality constraints
        """

        data_path = os.path.join('datasets', 'csv_tables')
        database = ParamsHandler.load_parameters('database')
        modality_wise_datasets = database['modality_wise_datasets']
        plog_threshold = ParamsHandler.load_parameters('settings')['eye_tracking_calibration_flag']

        pids_diag = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))['interview']

        # for single_task mode, all PIDs inside should be intersected with each other based on which modalities are True
        # make a dict with the key being mode, and the value being a modular function that works with different modes
        pids_mod = []
        for modality in modalities:
            task_mod = task
            filename = modality_wise_datasets[modality]

            # for eye modality, PIDs from eye_fixation and participant_log are intersected
            if modality == 'eye':
                table_eye = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_eye = table_eye.loc[table_eye['task'] == task_mod]['interview']

                table_plog = pd.read_csv(os.path.join(data_path, filename[1]))
                pids_plog = table_plog[table_plog['Eye-Tracking Calibration?'] >= plog_threshold]['interview']

                pids_mod.append(np.intersect1d(pids_eye, pids_plog))

            # for speech modality the files being accessed (text and audio) have tasks as 1, 2, 3 under the tasks column
            # then PIDs from text and audio are intersected
            if modality == 'speech':
                task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None

                table_audio = pd.read_csv(os.path.join(data_path, filename[0]))  # add support for more than one filenames like for speech
                pids_audio = table_audio.loc[table_audio['task'] == task_mod]['interview']

                table_text = pd.read_csv(os.path.join(data_path, filename[1]))
                pids_text = table_text[table_text['task'] == task_mod]['interview']

                pids_mod.append(np.intersect1d(pids_audio, pids_text))

            # PIDs from moca are used
            if modality == 'moca':
                table_moca = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_moca = table_moca['interview']
                pids_mod.append(pids_moca)

            # PIDs from mm_overall are used
            if modality == 'multimodal':
                table_multimodal = pd.read_csv(os.path.join(data_path, filename[0]))
                pids_multimodal = table_multimodal.loc[table_multimodal['task'] == task_mod]['interview']
                pids_mod.append(pids_multimodal)

        # for single task mode, we require an intersection of all PIDs, from all modalities
        if mode == 'single_tasks':
            while len(pids_mod) > 1:
                pids_mod = [np.intersect1d(pids_mod[i], pids_mod[i+1]) for i in range(len(pids_mod) - 1)]

        # for fusion mode, we require a union of PIDs taken from each modality (which were intersected internally within a modality)
        elif mode == 'fusion':
            while len(pids_mod) > 1:
                pids_mod = [np.union1d(pids_mod[i], pids_mod[i+1]) for i in range(len(pids_mod) - 1)]

        # intersecting the final list of PIDs with diagnosis, to get the PIDs with valid diagnosis
        pids = list(np.intersect1d(pids_mod[0], pids_diag))

        return pids

    @staticmethod
    def get_data(task: str, feature_set: dict, pids: list) -> List:
        feature_path = os.path.join('feature_sets')
        sub_feature_path = os.path.join(feature_path, 'sub_feature_sets')
        final_features = []
        initial_dataset = pd.DataFrame(pids, columns=['interview'])

        # checks if there are multiple features in a list under either eye or language (like there is Audio and Text under lang for Memory)
        features = list(feature_set.values())[0] if type(list(feature_set.values())[0]) == list else list(feature_set.values())

        # unpacking all features from their feature sets into final_features
        for feat in features:
            sub_features = ParamsHandler.load_parameters(os.path.join(feature_path, feat))
            final_features.extend(sub_features)

        # getting the list of features that are to be selected from the datasets
        to_select = []
        for feat in final_features:
            # currently cannot get sub_features for ET based features since they're in different files. but can do the rest of language based features
            # this would require copying feature names from the ET dataset files and putting them in sub_feature files like the lang ones
            # to make this portion work for everything, the Eye based datasets should be combined

            # language based to_select sub_features would exist in a merged version of Text and Audio datasets.
            # eye based to_select should then be different, so as to get them from the combined Eye dataset file (TO:DO)

            to_select.extend(ParamsHandler.load_parameters(os.path.join(sub_feature_path, feat)))

            # after getting to_select values, get the data from dataset files
            # we could first get the data then merge them, or merge them first then get the data, whichever would be less time consuming
            data_path = os.path.join('datasets', 'csv_tables')

            # if to_select has any features in it, then merge text and acoustic datasets, then choose the columns in the merged set that corresponds to
            # the features in to_select, for the task specified in the beginning
            def function_here:
                if len(to_select) > 0:
                    task_mod = 1 if task == 'CookieTheft' else 2 if task == 'Reading' else 3 if task == 'Memory' else None

                    text_data = pd.read_csv(os.path.join(data_path, 'text.csv'))
                    acoustic_data = pd.read_csv(os.path.join(data_path, 'acoustic.csv'))

                    lang_merged = pd.merge(text_data, acoustic_data, on=['interview', 'task'])
                    lang_dataset = lang_merged.loc[lang_merged['task'] == task_mod]

                    to_select.append('interview')
                    lang_dataset = lang_dataset[to_select]
                    dataset = pd.merge(initial_dataset, lang_dataset, on='interview')






        return []
