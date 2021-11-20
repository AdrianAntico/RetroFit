# QA
import pkg_resources
import timeit
import datatable as dt
from datatable import f
import retrofit
from retrofit import DatatableViz as dtv
import numpy as np
import plotly.express as px
import plotly.io as pio


# No Group Example: datatable
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)

# No marginal plots
x = dtv.ScatterPlot(
  data = data,
  N = 10000,
  XVar = 'XREGS1',
  YVar = 'Leads',
  FacetColVar = None,
  FacetRowVar = None,
  ColorVar = 'Leads',
  SizeVar = 'Leads',
  SymbolVar = None,
  HoverStatsVar = None,
  MarginalX = 'histogram',
  MarginalY = 'histogram')

# With marginal plots
x = dtv.ScatterPlot(
  data = data, 
  N = 10000, 
  XVar = 'XREGS1', 
  YVar = 'Leads', 
  FacetColVar = 'MarketingSegments',  
  FacetRowVar = 'MarketingSegments2',
  ColorVar = 'Leads',
  SizeVar = 'Leads', 
  SymbolVar = None,
  HoverStatsVar = None, 
  MarginalX = 'histogram',
  MarginalY = None)


# Args
# N = 10000
# XVar = 'XREGS1'
# YVar = 'Leads'
# FacetColVar = 'MarketingSegments'
# FacetRowVar = 'MarketingSegments2'
# ColorVar = 'Leads'
# SizeVar = 'Leads'
# SymbolVar = None
# HoverStatsVar = None
# MarginalX = 'histogram'
# MarginalY = None
