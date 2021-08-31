# QA: Test FE0_AutoLags
import timeit
import datatable as dt
import polars as pl
import retrofit
from retrofit import FeatureEngineering as fe

## No Group Example: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data1.names)
print(ArgsList)

# # Args
# ArgsList=None
# LagPeriods=1
# LagColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=None
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

## No Group Example: polars
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data2.columns)
print(ArgsList)

# # Args
# data=data
# LagPeriods=1
# LagColumnNames='Weekly_Sales'
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# ImputeValue=-1.0
# Sort=True
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'

## Group Example, Single Lag: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data1.names)
print(ArgsList)

# # Args
# ArgsList=None
# LagPeriods=1
# LagColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

## Group Exmaple: polars (Impute = -1 is failing, RuntimeError: Any(Other("Cannot cast list type")))
# Issue raised #11224
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data2.columns)
print(ArgsList)

# # Args
# ArgsList=None
# LagPeriods=1
# LagColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# ImputeValue=-1.0
# Sort=True
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'


## Group and Multiple Periods and LagColumnNames: datatable
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data1 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data1.names)
print(ArgsList)

# # Args
# ArgsList=None
# LagPeriods=[1,3,5]
# LagColumnNames=['Leads','XREGS1']
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

## Group and Multiple Periods and LagColumnNames: datatable
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoLags(data=data, ArgsList=None, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], ImputeValue=-1.0, Sort=True, Processing='polars', InputFrame='polars', OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data2 = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data2.columns)
print(ArgsList)

# # Args
# ArgsList=None
# LagPeriods=[1,3,5]
# LagColumnNames=['Leads','XREGS1']
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# ImputeValue=-1.0
# Sort=True
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'

#########################################################################################################
#########################################################################################################

# QA FE0_AutoRollStats
import timeit
import datatable as dt
import polars as pl
from retrofit import FeatureEngineering as fe

## No Group Example
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoRollStats(data=data, ArgsList=None, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# RollColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=None
# MovingAvg_Periods=[3,5,7]
# MovingSD_Periods=[3,5,7]
# MovingMin_Periods=[3,5,7]
# MovingMax_Periods=[3,5,7]
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

## No Group Example
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoRollStats(data=data, ArgsList=None, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# RollColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=None
# MovingAvg_Periods=[3,5,7]
# MovingSD_Periods=[3,5,7]
# MovingMin_Periods=[3,5,7]
# MovingMax_Periods=[3,5,7]
# ImputeValue=-1
# Sort=True
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'


## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoRollStats(data=data, ArgsList=None, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# RollColumnNames=['Leads','XREGS1']
# DateColumnName='CalendarDateColumn'
# ByVariables=['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# MovingAvg_Periods=[3,5,7]
# MovingSD_Periods=[3,5,7]
# MovingMin_Periods=[3,5,7]
# MovingMax_Periods=[3,5,7]
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoRollStats(data=data, ArgsList=None, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# RollColumnNames='Leads'
# DateColumnName='CalendarDateColumn'
# ByVariables=None
# MovingAvg_Periods=[3,5,7]
# MovingSD_Periods=[3,5,7]
# MovingMin_Periods=[3,5,7]
# MovingMax_Periods=[3,5,7]
# ImputeValue=-1
# Sort=True
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

#########################################################################################################
#########################################################################################################

# QA FE0_AutoDiff
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
    
## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label'], DiffNumericVariables = 'Leads', DiffCalendarDateColumnVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# DateColumnName = 'CalendarDateColumn'
# ByVariables = ['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# DiffNumericVariables = 'Leads'
# DiffCalendarDateColumnVariables = 'CalendarDateColumn'
# DiffGroupVariables = None
# NLag1 = 0
# NLag2 = 1
# Sort=True
# Processing = 'datatable'
# InputFrame = 'datatable'
# OutputFrame = 'datatable'

## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label'], DiffNumericVariables = 'Leads', DiffCalendarDateColumnVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# DateColumnName = 'CalendarDateColumn'
# ByVariables = ['MarketingSegment','MarketingSegment2','MarketingSegment3', 'Label']
# DiffNumericVariables = 'Leads'
# DiffCalendarDateColumnVariables = 'CalendarDateColumn'
# DiffGroupVariables = None
# NLag1 = 0
# NLag2 = 1
# Sort=True
# Processing = 'datatable'
# InputFrame = 'datatable'
# OutputFrame = 'datatable'

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffCalendarDateColumnVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']
del Output
print(data.names)
print(ArgsList)

# # Args
# ArgsList=None
# DateColumnName = 'CalendarDateColumn'
# ByVariables = None
# DiffNumericVariables = 'Leads'
# DiffCalendarDateColumnVariables = 'CalendarDateColumn'
# DiffGroupVariables = None
# NLag1 = 0
# NLag2 = 1
# Sort=True
# Processing = 'datatable'
# InputFrame = 'datatable'
# OutputFrame = 'datatable'

#########################################################################################################
#########################################################################################################

# FE1_AutoCalendarVariables
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
Output = fe.FE1_AutoCalendarVariables(data=data, ArgsList=None, DateColumnNames = 'CalendarDateColumn', CalendarVariables = ['wday','mday','wom','month','quarter','year'], Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)

#########################################################################################################
#########################################################################################################

import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
Output = fe.FE1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
data = Output['data']
ArgsList = Output['ArgsList']

#########################################################################################################
#########################################################################################################

# FE2_AutoDataParition
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import utils as u
    
# random
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
DataSets = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName='CalendarDateColumn', 
  PartitionType='random', 
  Ratios=[0.70,0.20,0.10], 
  ByVariables=None, 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
ArgsList = DataSets['ArgsList']
