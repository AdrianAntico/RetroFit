# QA: Test AutoLags
import timeit
import datatable as dt
import polars as pl
from retrofit import TimeSeriesFeatures as ts
import pandas as pd

## No Group Example: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/RemixAutoML/tests/QA_DataSets/ThreeGroup-FC-Walmart.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=1, LagColumnNames='Weekly_Sales', DateColumnName='Date', ByVariables=None, ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

## No Group Example: polars
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/RemixAutoML/tests/QA_DataSets/ThreeGroup-FC-Walmart.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=1, LagColumnNames='Weekly_Sales', DateColumnName='Date', ByVariables=None, ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
data=data
LagPeriods=1
LagColumnNames='Weekly_Sales'
DateColumnName='Date'
ByVariables=['Region','Store','Dept']
ImputeValue=-1.0
Sort=True
Processing='polars'
InputFrame='polars'
OutputFrame='polars'

t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.columns)
print(ArgsList)

# Check equality
data1 = data1.to_pandas()
data2 = data2.to_pandas()



## Group Example, Single Lag: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='Date', ByVariables=['Region','Store','Dept'], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data1.names)
print(ArgsList)

## Group Exmaple: polars (Impute = -1 is failing, RuntimeError: Any(Other("Cannot cast list type")))
# Issue raised #11224
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='Date', ByVariables=['Region','Store','Dept'], ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data2.columns)
print(ArgsList)

# Check if equal
data1 = data1.to_pandas()
data2 = data2.to_pandas()


## Group and Multiple Periods and LagColumnNames: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='Date', ByVariables=['Region','Store','Dept'], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data1.names)
print(ArgsList)

## Group and Multiple Periods and LagColumnNames: datatable
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='Date', ByVariables=['Region','Store','Dept'], ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data2.columns)
print(ArgsList)

# Check if equal
data1 = data1.to_pandas()
data2 = data2.to_pandas()
data1.equals(data2)


#########################################################################################################

# QA AutoRollStats
import datatable as dt
from datatable import sort, f, by

## No Group Example
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='Date', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='Date', ByVariables=['Region','Store','Dept'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='Date', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)


# QA AutoDiff
import datatable as dt
from datatable import sort, f, by
    
## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoDiff(data=data, DateColumnName = 'Date', ByVariables = ['Region','Store','Dept'], DiffNumericVariables = 'Leads', DiffDateVariables = 'Date', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoDiff(data=data, DateColumnName = 'Date', ByVariables = ['Region','Store','Dept'], DiffNumericVariables = 'Leads', DiffDateVariables = 'Date', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = ts.AutoDiff(data=data, DateColumnName = 'Date', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'Date', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)
