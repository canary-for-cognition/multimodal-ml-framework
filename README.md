# multimodal-ml-framework
 
The updated Machine Learning framework for the CANARY project. This framework utilizes eye-tracking data and speech data to predict if a participant is a patient or a healthy control. 

## Setup
1. (Recommended) Install Anaconda 3 on your system: https://www.anaconda.com/products/individual
2. Download/clone this repository wherever you'd like in your system.
3. Open a terminal **within the repository folder**. In the terminal (Linux)/Command Prompt or Anaconda Prompt (Windows), execute this command: 
   `conda create --name canary python=3.8.5` .
   This will create a virtual environment called "canary", so that it doesn't mess up your normal installation environment.
4. To activate this environment, execute in the same terminal:
   `conda activate canary`. 
5. To install all the dependencies of this framework, execute:
   `pip install -r requirements.txt`
6. (Optional) If you make any changes to the python packages and would like to save them in the requirements.txt file, then execute:
   `pip freeze requirements.txt`
   
## Editing parameters
To use the framework, the file **params/settings.yaml** contains all the parameters to run the experiment. All the parameters defined in **params/settings.yaml** are modifiable, but the ones that would need to be changed the most depending on each experiment are:
1. **seeds**: defines the number of seeds the framework will run for
2. **folds**: defines number of folds for CrossValidation
3. **mode**: defines the mode of experiment. Can be _single_tasks_ or can be _fusion_. _ensemble_ will be added soon.
    1. _single_tasks_: runs each of the specified tasks and reports their results separately. 
    2. _fusion_: runs all the specified tasks separately, then combines their results through averaging and reports results.
4. **classifiers**: a list of classifiers that can be run through the framework. 
5. **tasks**: a list of tasks that can be run through the framework. 
6. **output_folder**: name of the folder where the results will be stored at the end of the run. This folder will be kept under the **results** folder.

Within **params/settings.yaml** file, all possible input values for each parameter have been provided in the comments of the file.

## Running the framework
After setting up the parameters, it's time to run the framework. Either use a Python IDE and open **main.py** and run it from there, or using the terminal, execute this line within the repository folder:
`python main.py`
Note: The canary environment has to be activated everytime a new terminal is opened.

The results would be saved at **results/output_folder** where the output_folder is the one specified in **params/settings.yaml**.

## Generating results over all the seeds
The file **compile_results.py** is used to compile the results across all seeds and creates a cleaner looking table.
To compile the results across all seeds, open the terminal and execute:

`python compile_results.py <path to output folder> <path to file and name of file>`

_For example:_ for compiling results under a folder **TF**, this above line would be:

`python compile_results.py ./results/TF ./results/TF/tf`, 
where **./results/TF/tf** is the path where the compiled results should be saved, and **tf** is the name of the file. 
The resulting file is a .csv file. No need to specify .csv in the filename for the above line.

