# Module: utils
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

import pickle
import os


def do_call(FUN, args=None, kwargs=None):
    """
    Mimic R's do.call() functionality in Python.
    
    Parameters:
        FUN:    The function to be called.
        args:   A list or tuple of positional arguments.
        kwargs: A dictionary of keyword arguments.
    
    Returns:
        The result of calling FUN with the provided arguments.
    
    Example:
        def my_func(alpha, beta):
            return alpha + beta
        
        result = do_call(my_func, args=[0.01], kwargs={'beta': 1})
        print(result)  # Outputs: 1.01
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return FUN(*args, **kwargs)


def save(x, Path):
    """
    Save a Python object to a file using pickle, similar in spirit to R's save().
    
    Parameters:
        x (any): The object you wish to save.
        Path (str): The file path (including file name). The .pkl extension will be appended
                    if not already present.
    
    Raises:
        ValueError: If Path is not provided.
    
    Example:
        my_object = {"alpha": 0.01, "beta": 1}
        save(my_object, "my_saved_object")
    """
    if Path is None:
        raise ValueError("A valid file path must be provided.")
    
    # Ensure the file name ends with .pkl
    if not Path.endswith('.pkl'):
        Path = f"{Path}.pkl"
    
    try:
        with open(Path, 'wb') as out:
            pickle.dump(x, out, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"An error occurred while saving the object: {e}")


def load(Path):
    """
    Load a Python object from a pickle file, mimicking R's load() functionality.
    
    Parameters:
        Path (str): The file path (including file name). If the path doesn't end with '.pkl',
                    the function will first check for a file with the '.pkl' extension.
    
    Returns:
        The Python object loaded from the pickle file.
    
    Raises:
        ValueError: If no Path is provided.
        FileNotFoundError: If the file is not found.
    """
    if Path is None:
        raise ValueError("A valid file path must be provided.")
    
    # If the provided path doesn't end with '.pkl', try appending it if the file exists.
    if not Path.endswith('.pkl'):
        potential_path = f"{Path}.pkl"
        if os.path.exists(potential_path):
            Path = potential_path
    
    if not os.path.exists(Path):
        raise FileNotFoundError(f"The file '{Path}' does not exist.")
    
    try:
        with open(Path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        raise


def cumsum(x):
    """
    Create a list of summed up values from another list
    """
  
    # Initialize list with prefilled values
    AccumRatios = [z for z in range(len(x))]
  
    # Fill accumulation list
    for acc in range(len(x)):
        if acc == 0:
            AccumRatios[acc] = x[acc]
        else:
            AccumRatios[acc] = AccumRatios[acc-1] + x[acc]

    # Return list
    return(AccumRatios)


def print_dict(x):
    """
    Print out the dictionary where each key : value pair gets a new line
    """
    for z in x:
        if not x[z] is None:
            print(z + ': ' + str(x[z]))
        else:
            print(z + ': None')
