# Module: utils
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

from retrofit import utils
import pickle


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


def printdict(x):
    """
    Print out the dictionary where each key : value pair gets a new line
    """
    for z in x:
        if not x[z] is None:
            print(z + ': ' + str(x[z]))
        else:
            print(z + ': None')


def do_call(FUN, args=[], kwargs = {}):
    """
    # Goal
    Run code like R do.call
    
    # Parameters
    FUN:    a function you'd like to run
    args:   vector element args
    kwargs: dictionary args
    
    # Example
    ArgsList = dict()
    ArgsList['alpha'] = 0.01
    ArgsList['beta'] = 1
    ArgsList['beta'] = 1
    do_call()
    
    """
    return FUN(*args, **kwargs)

def save(x = None, Path = None):
    """
    # Goal:
    Save python objects in a similar way to saving R objects in R
  
    # Output
    Save objects to file
  
    # Parameters
    x:    Object you wish to save
    Path: File path + file name + extension
    """
    with open(f"{Path}.pkl", 'wb') as out:
        pickle.dump(x, out, pickle.HIGHEST_PROTOCOL)


def load(Path):
    """
    # Goal:
    Load python objects in a similar way to loading R objects in R
  
    # Output
    Save objects to file
  
    # Parameters
    x:    Object you wish to save
    Path: File path + file name + extension
    """
    with open(f"{Path}", 'rb') as x:
        return pickle.load(x)
