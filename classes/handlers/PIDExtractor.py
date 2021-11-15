import numpy as np
import pandas as pd
import os

from classes.handlers.ParamsHandler import ParamsHandler


class PIDExtractor:
    def __init__(self, mode: str, extraction_method: str, pid_file_paths: dict, dataset_name: str):
        self.__dataset_name = dataset_name
        supp_datasets = ["canary", "dementia_bank"]
        if self.__dataset_name not in supp_datasets:
            raise ValueError("Dataset '{}' is not supported! Supported datasets are: {}"
                             .format(self.__dataset_name, supp_datasets))

        self.__mode = mode
        self.__extraction_method = extraction_method
        self.__pid_file_paths = pid_file_paths
        self.__superset_ids = []

    @staticmethod
    def __fetch_eye_pids(task: str, data_path: str, filename: str, plog_thr: float) -> tuple:
        table_eye = pd.read_csv(os.path.join(data_path, filename[0]))
        pids_eye = table_eye.loc[table_eye['task'] == task]['interview']
        table_plog = pd.read_csv(os.path.join(data_path, filename[1]))
        pids_plog = table_plog[table_plog['Eye-Tracking Calibration?'] >= plog_thr]['interview']
        return np.intersect1d(pids_eye, pids_plog)

    def __fetch_speech_pids(self, task: str, data_path: str, filename: str) -> tuple:
        if self.__dataset_name == 'canary':
            task = {'CookieTheft': 1, 'Reading': 2, 'Memory': 3}[task]

            table_audio = pd.read_csv(os.path.join(data_path, filename[0]))
            pids_audio = table_audio.loc[table_audio['task'] == task]['interview']

            table_text = pd.read_csv(os.path.join(data_path, filename[1]))
            pids_text = table_text[table_text['task'] == task]['interview']

            return np.intersect1d(pids_audio, pids_text)

        if self.__dataset_name == 'dementia_bank':
            dbank_pids_all = [pd.read_csv(os.path.join(data_path, i))['interview'] for i in filename]
            while len(dbank_pids_all) > 1:
                dbank_pids_all = [np.intersect1d(dbank_pids_all[i], dbank_pids_all[i + 1])
                                  for i in range(len(dbank_pids_all) - 1)]
            return dbank_pids_all[0]

    @staticmethod
    def __fetch_moca_pids(data_path: str, filename: str) -> tuple:
        return pd.read_csv(os.path.join(data_path, filename[0]))['interview']

    @staticmethod
    def __fetch_multimodal_pids(task: str, data_path: str, filename: str) -> tuple:
        table_multimodal = pd.read_csv(os.path.join(data_path, filename[0]))
        return table_multimodal.loc[table_multimodal['task'] == task]['interview']

    def __fetch_modality_pids(self, task: str, modality: str, filename: str, data_path: str, plog_thr: float) -> list:
        pids_mod = []

        # For eye modality, PIDs from eye_fixation and participant_log are intersected
        if modality == 'eye':
            pids_mod.append(self.__fetch_eye_pids(task, data_path, filename, plog_thr))

        # For speech modality the files being accessed (text and audio) have tasks as 1, 2, 3 under the tasks column
        # then PIDs from text and audio are intersected
        if modality == 'speech':
            pids_mod.append(self.__fetch_speech_pids(task, data_path, filename))

        # PIDs from MOCA are used
        if modality == 'moca':
            pids_mod.append(self.__fetch_moca_pids(data_path, filename))

        # PIDs from mm_overall are used
        if modality == 'multimodal':
            pids_mod.append(self.__fetch_multimodal_pids(task, data_path, filename))

        return pids_mod

    def __combine_pids(self, pids_mod: list) -> list:
        while len(pids_mod) > 1:
            # for single task and ensemble modes, we require an intersection of all PIDs, from all modalities
            if self.__mode == 'single_tasks':
                pids_mod = [np.intersect1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]

            # for fusion mode, we require a union of PIDs taken from each modality
            elif self.__mode == 'fusion' or self.__mode == 'ensemble':
                pids_mod = [np.union1d(pids_mod[i], pids_mod[i + 1]) for i in range(len(pids_mod) - 1)]
        return pids_mod

    def __fetch_task_pids(self, task: str) -> list:
        """
        :param task: the task for which PIDs are required
        :return: list of PIDs that satisfy the task and modality constraints
        """

        modalities = ParamsHandler.load_parameters(os.path.join(self.__dataset_name, task))['modalities']

        database = ParamsHandler.load_parameters(os.path.join(self.__dataset_name, 'database'))
        modality_wise_datasets = database['modality_wise_datasets']
        data_path = os.path.join('datasets', self.__dataset_name)
        plog_threshold = ParamsHandler.load_parameters('settings')['eye_tracking_calibration_flag']

        pids_mod = []
        for modality in modalities:
            filename = modality_wise_datasets[modality]
            pids_mod += (self.__fetch_modality_pids(task, modality, filename, data_path, plog_threshold))

        pids_mod = self.__combine_pids(pids_mod)

        # Intersecting the final list of PIDs with diagnosis, to get the PIDs with valid diagnosis
        pids_diag = pd.read_csv(os.path.join(data_path, 'diagnosis.csv'))['interview']
        pids = list(np.intersect1d(pids_mod[0], pids_diag))

        return pids

    def extract_pids(self, tasks: list):
        superset_ids = []
        for task in tasks:
            # Getting pids and saving them at pid_file_path for each task
            pids = self.__fetch_task_pids(task=task)
            pd.DataFrame(pids, columns=['interview']).to_csv(self.__pid_file_paths[task])
            superset_ids.append(pids)

        if self.__mode == 'fusion' or self.__mode == 'ensemble':
            # Getting superset_ids for fusion, which are the union of all lists of PIDs taken from all tasks
            while (len(superset_ids)) > 1:
                superset_ids = [np.union1d(superset_ids[i], superset_ids[i + 1]) for i in range(len(superset_ids) - 1)]

            self.__superset_ids = superset_ids[0]
            file_name = "{}_{}_super_pids.csv".format(self.__mode, self.__extraction_method)
            super_pids_file_path = os.path.join('assets', self.__dataset_name, 'PIDs', file_name)

            print("\n\t\t --> * Created superset of PIDs for mode '{}' * ".format(self.__mode))
            pd.DataFrame(self.__superset_ids, columns=['interview']).to_csv(super_pids_file_path)
