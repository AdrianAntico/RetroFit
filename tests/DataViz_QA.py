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

dtv.ScatterPlot(data, XVar='XREGS1', YVar='Leads', GroupVariables=None)
 

dtv.ScatterPlot(data, XVar='XREGS1', YVar='Leads', GroupVariables='MarketingSegments')

  
XVar='XREGS1'
YVar='Leads'
GroupVariables='MarketingSegments'
ColorVar='Leads'
SizeVar='XREGS1'
HoverStatsVar='Leads'

ScatterPlot(data=data, XVar='XREGS1', YVar='Leads', ColorVar='Leads', SizeVar='XREGS', HoverStatsVar='Leads', GroupVariables=None)


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()
