# Module: utils
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.0
# Last modified : 2021-09-03

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
