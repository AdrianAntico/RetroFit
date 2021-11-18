# QA: Test FE0_AutoLags
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import PolarsFE as pfe

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# No Group Example: datatable
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)


data


 scattplot(data, XVar='XREGS1', YVar='Leads', GroupVariables=None)
 

 scattplot(data, XVar='XREGS1', YVar='Leads', GroupVariables='MarketingSegments')
  
 
