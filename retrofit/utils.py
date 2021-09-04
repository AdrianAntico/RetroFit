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
