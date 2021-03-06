# QA Plotly
import pkg_resources
import timeit
import datatable as dt
from datatable import f
import retrofit
from retrofit import PlotlyViz as pv
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars
import pandas


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ScatterPlots
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# No Group Example: datatable
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')

# Pull in data
#data = dt.fread(FilePath)
data = polars.read_csv(FilePath)
data = pandas.read_csv(FilePath)

# No marginal plots
x = pv.ScatterPlot(
  data = data,
  Frame = 'datatable',
  Title = 'Adrian is the best',
  N = 10000,
  XVar = 'XREGS1',
  YVar = 'Leads',
  FacetColVar = None, 
  FacetColMaxLevels=None,
  FacetRowVar = None,
  FacetRowMaxLevels=None,
  ColorVar = 'Leads',
  SizeVar = 'Leads',
  SymbolVar = None,
  HoverStatsVar = None,
  MarginalX = 'histogram',
  MarginalY = 'histogram',
  TrendLine = 'ols',
  Copula = False,
  XLim = None,
  YLim = None)

# With marginal plots
x = pv.ScatterPlot(
  data = data, 
  Frame = 'datatable',
  Title = 'Kiran sucks dick',
  N = 10000, 
  XVar = 'XREGS1', 
  YVar = 'Leads', 
  FacetColVar = 'MarketingSegments',  
  FacetRowVar = 'MarketingSegments2',
  FacetColMaxLevels=2,
  FacetRowMaxLevels=2,
  ColorVar = 'Leads',
  SizeVar = 'Leads', 
  SymbolVar = None,
  HoverStatsVar = None, 
  MarginalX = 'histogram',
  MarginalY = None,
  TrendLine = 'ols',
  Copula =  False,
  XLim = [0, 0.60],
  YLim = [0, 0.40])


# Args
# import numpy as np
# import plotly.express as px
# import datatable as dt
# from datatable import f, sort, update
# import plotly.io as pio
# N = 10000
# XVar = 'XREGS1'
# YVar = 'Leads'
# FacetColVar = 'MarketingSegments'
# FacetColMaxLevels=2
# FacetRowVar = 'MarketingSegments2'
# FacetRowMaxLevels=2
# ColorVar = 'Leads'
# SizeVar = 'Leads'
# SymbolVar = None
# HoverStatsVar = None
# MarginalX = 'histogram'
# MarginalY = None
# TrendLine = 'ols'
# Copula = True
# Title='aa'
# XLim = [0, 0.60]
# YLim = [0, 0.40]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# LinePlots
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# BarPlots
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# BoxPlots
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





