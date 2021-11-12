# finding-similar-items-textually-similar-documents

## Setup

Make a conda virtual environment from the ```environment.yml``` file as discussed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Make the virtual environment available in Jupyter Notebooks as discussed [here](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook). Start Jupyter Notebooks and select the environment. Run the ```main.ipynb``` notebook.

## Reproducibility

The Python ```PYTHONHASHSEED``` environment variable is fixed so that the built-in Python ```hash()``` function yields consistent results. Pass the ```seed``` variable to the ```MinHashing``` class constructor as minhashing uses the ```numpy.random.randint()``` function.

## Results

