from classes.cv.CrossValidator import CrossValidator
from classes.handlers.DataHandler import DataHandler
from classes.handlers.ParamsHandler import ParamsHandler
from classes.handlers.ResultsHandler import ResultsHandler

import os
from multiprocessing import Process, Pool
import warnings

warnings.filterwarnings("ignore")


def main():
    # load_parameters to take the name of settings file (without .yaml extension)
    params = ParamsHandler.load_parameters('settings')
    seeds = params["seeds"]
    mode = params["mode"]
    tasks = params["tasks"]
    classifiers = params["classifiers"]
    output_folder = params["output_folder"]
    extraction_method = params["PID_extraction_method"]
    dataset_name = params['dataset']

    path = os.path.join(os.getcwd(), 'results', dataset_name, output_folder)
    if not os.path.exists(path):
        os.mkdir(path)

    # Getting the data from DataHandler and models from ModelsHandler
    tasks_data = DataHandler(mode, extraction_method).load_data(tasks=tasks)

    # Running CrossValidator on the extracted data for the number of seeds specified
    # multiprocessing - change number of cpu cores to use based on preference
    cpu_count = os.cpu_count()
    pool = Pool(processes=cpu_count)
    cv = CrossValidator(mode, classifiers)
    cv = [pool.apply_async(cv.cross_validate, args=(seed, tasks_data)) for seed in range(seeds)]
    _ = [p.get() for p in cv]

    # Compile results over all seeds into a single output table
    # the name of the file will be the same as the results folder name specified in settings.yaml
    ResultsHandler.compile_results(dataset_name, output_folder)


if __name__ == '__main__':
    main()
