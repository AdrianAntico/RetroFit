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
