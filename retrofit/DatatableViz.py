# Module: DatatableViz
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

import plotly.express as px
import datatable as dt
import plotly.io as pio

def ScatterPlot(data, XVar=None, YVar=None, GroupVariables=None):
  """
  # Goal:
  Automatically generate scatterplots from datatable data

  # Output
  Return a plot object and print to screen

  # Parameters
  data: Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
  XVar: Column name of the variable to use for the x-axis
  YVar: Column name of the variable to use for the y-axis
  """
  
  # Ensure datatable
  if not isinstance(data, dt.Frame):
      raise Exception("data needs to be a datatable frame")
 
  
  # Ensure XVar is not None
  if not XVar:
      raise Exception("XVar cannot be None")

  # Ensure XVar is not None
  if not YVar:
      raise Exception("YVar cannot be None")

  # Define x, y, ByList for plotting by converting to lists
  x = data[:, XVar].to_list()[0]
  y = data[:, YVar].to_list()[0]
  
  # Build config list
  if not GroupVariables:
      config = [dict(
        type = 'scatter',
        x = x,
        y = y,
        mode = 'markers')]
    
  else:  
  
    GroupVariables = data[:, GroupVariables].to_list()
    GroupVariables = list(set(GroupVariables[0]))
  
    # TODO: add more color options for cases with high cardinality GroupVariable levels
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colors=['b','g','r','c','m','y','k']
    Style = []
    for i in range(len(GroupVariables)):
      Style.append(dict(target = GroupVariables[i], value = dict(marker = dict(color = colors[i]))))

    # Configure
    config = [dict(
      type = 'scatter',
      x = x,
      y = y,
      mode = 'markers',
      color = GroupVariables]
  
  # Create and show plot
  fig_dict = dict(data=config)
  pio.show(fig_dict, validate=False)
