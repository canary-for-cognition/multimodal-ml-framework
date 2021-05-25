from classes.CrossValidator import CrossValidator
from classes.handlers.DataHandler import DataHandler
from classes.handlers.ModelsHandler import ModelsHandler
from classes.handlers.ParamsHandler import ParamsHandler


def main():
    params = ParamsHandler.load_parameters()
    mode = params["mode"]
    tasks = params["tasks"]
    classifiers = params["models"]

    tasks_data = DataHandler.load_data(tasks, mode)
    models = ModelsHandler.get_models(classifiers)

    results = []
    for seed in range(params["seeds"]):
        print("\nProcessing seed {}\n".format(seed))

        """
        Single tasks
        * Each classifier process data stemming from each individual tasks
        * The output is a prediction for each task and classifier
        """
        if mode == "single_tasks":
            for task_data in tasks_data:
                for model in models:
                    metrics = CrossValidator.cross_validate(model, task_data)
                    results.append(metrics)

        """
        Task fusion
        * Each classifier process data stemming from all tasks at the same time
        * Individual classifiers are built for each modality using the same type of classifier
        * The individual task predictions are merged via averaging/stacking
        * The output is a prediction for each classifier
        """
        if mode == "fusion":
            for model in models:
                metrics = CrossValidator.cross_validate(model, tasks_data)
                results.append(metrics)

        """
        Models ensemble
        * For each task, all classifiers make a prediction on task data.
        * The individual classifiers predictions are merged via averaging/stacking
        * The output is a prediction for each task
        """
        if mode == "ensemble":
            for task_data in tasks_data:
                metrics = CrossValidator.cross_validate(models, task_data)
                results.append(metrics)


if __name__ == '__main__':
    main()
