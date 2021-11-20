# QA: Test FE0_AutoLags
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableViz as dtv


# No Group Example: datatable
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)

dtv.ScatterPlot(
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
  Markers = None)
  
