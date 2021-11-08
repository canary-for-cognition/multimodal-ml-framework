import os
import pandas as pd

METRICS = 'metric'
MODEL = 'model'
ACCURACY = 'acc'
ROC = 'roc'
F1_SCORE = 'f1'
PRECISION = 'precision'
RECALL = 'recall'
SETTINGS = 'settings'
SPECIFICITY = 'specificity'

ACCURACY_SD = 'acc_sd'
ROC_SD = 'roc_sd'
F1_SD = 'f1_sd'
PREC_SD = 'prec_sd'
REC_SD = 'rec_sd'
SPEC_SD = 'spec_sd'

RESULT_COLUMNS = [SETTINGS, MODEL, ACCURACY, ROC, F1_SCORE, PRECISION, RECALL, SPECIFICITY]
RESULT_COLUMNS2 = [SETTINGS, MODEL, ACCURACY, ACCURACY_SD, ROC, ROC_SD, F1_SCORE, F1_SD, PRECISION, PREC_SD,
                   RECALL, REC_SD, SPECIFICITY, SPEC_SD]


class ResultsHandler:
    def __init__(self):
        pass

    @staticmethod
    def compile_results(dataset_name: str, foldername: str):
        input_files = os.path.join(os.getcwd(), 'results', dataset_name, foldername)
        results_csv = pd.DataFrame(columns=RESULT_COLUMNS)
        for directory in os.listdir(input_files):
            if os.path.isdir(os.path.join(input_files, directory)):
                for filename in os.listdir(os.path.join(input_files, directory)):
                    if filename.startswith('results'):
                        
                        if len(filename[:-4].split('_')[-1]) == 0:
                            suffix = 'overall'
                        else:
                            suffix = ''
                        
                        results = pd.read_csv(os.path.join(input_files, directory, filename))
                        models = results.model.unique()

                        for model in models:
                            model_info = results[results[MODEL] == model]
                            
                            acc = model_info[model_info[METRICS] == ACCURACY]['1'].mean()
                            roc = model_info[model_info[METRICS] == ROC]['1'].mean()
                            f1_score = model_info[model_info[METRICS] == 'fms']['1'].mean()
                            precision = model_info[model_info[METRICS] == PRECISION]['1'].mean()
                            recall = model_info[model_info[METRICS] == RECALL]['1'].mean()
                            specificity = model_info[model_info[METRICS] == SPECIFICITY]['1'].mean()
                            
                            results_csv = results_csv.append({
                                SETTINGS: filename[:-4].split('_')[-1] + suffix,
                                MODEL: model,
                                ACCURACY: acc,
                                ROC: roc,
                                F1_SCORE: f1_score,
                                PRECISION: precision,
                                RECALL: recall,
                                SPECIFICITY: specificity
                            }, ignore_index=True)
        
        ResultsHandler.average_seeds(results_csv, dataset_name, foldername)

    @staticmethod
    def average_seeds(results: pd.DataFrame, dataset_name: str, foldername: str):
        results_csv = pd.DataFrame(columns=RESULT_COLUMNS)
        settings = results.settings.unique()
        for setting in settings:
            setting_groups = results[results[SETTINGS] == setting]
            models = results.model.unique()
            
            for model in models:
                setting_model_info = setting_groups[setting_groups[MODEL] == model]
                if setting_model_info.empty:
                    continue

                acc = round(setting_model_info[ACCURACY].mean(), 2)
                roc = round(setting_model_info[ROC].mean(), 2)
                f1_score = round(setting_model_info[F1_SCORE].mean(), 2)
                precision = round(setting_model_info[PRECISION].mean(), 2)
                recall = round(setting_model_info[RECALL].mean(), 2)
                specificity = round(setting_model_info[SPECIFICITY].mean(), 2)

                acc_sd = round(setting_model_info[ACCURACY].std(), 2)
                roc_sd = round(setting_model_info[ROC].std(), 2)
                f1_sd = round(setting_model_info[F1_SCORE].std(), 2)
                prec_sd = round(setting_model_info[PRECISION].std(), 2)
                rec_sd = round(setting_model_info[RECALL].std(), 2)
                spec_sd = round(setting_model_info[SPECIFICITY].std(), 2)

                pmi = u"\u00B1"
                # results_csv = results_csv.append({
                #     SETTINGS: setting,
                #     MODEL: model,
                #     ACCURACY: acc,
                #     ROC: roc,
                #     F1_SCORE: f1_score,
                #     PRECISION: precision,
                #     RECALL: recall,
                #     SPECIFICITY: specificity,
                #
                #     ACCURACY_SD: acc_sd,
                #     ROC_SD: roc_sd,
                #     F1_SD: f1_sd,
                #     PREC_SD: prec_sd,
                #     REC_SD: rec_sd,
                #     SPEC_SD: spec_sd
                # }, ignore_index=True)

                results_csv = results_csv.append({
                    SETTINGS: setting,
                    MODEL: model,
                    ACCURACY: str(acc) + pmi + str(acc_sd),
                    ROC: str(roc) + pmi + str(roc_sd),
                    F1_SCORE: str(f1_score) + pmi + str(f1_sd),
                    PRECISION: str(precision) + pmi + str(prec_sd),
                    RECALL: str(recall) + pmi + str(rec_sd),
                    SPECIFICITY: str(specificity) + pmi + str(spec_sd),
                }, ignore_index=True)
        
        outfile = os.path.join(os.getcwd(), 'results', dataset_name, foldername, foldername+'.csv')
        results_csv.to_csv(outfile, index=False, columns=RESULT_COLUMNS, encoding='utf-8-sig')

