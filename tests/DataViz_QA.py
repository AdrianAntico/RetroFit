# QA: Test FE0_AutoLags
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import PolarsFE as pfe
from retrofit import DatatableViz as dtv

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# No Group Example: datatable
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)


data

dtv.ScatterPlot()


dtv.ScatterPlot(data, XVar='XREGS1', YVar='Leads', GroupVariables=None)
 

dtv.ScatterPlot(data, XVar='XREGS1', YVar='Leads', GroupVariables='MarketingSegments')

  
XVar='XREGS1'
YVar='Leads'
GroupVariables='MarketingSegments'
