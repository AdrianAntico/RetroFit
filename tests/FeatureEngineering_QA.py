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

## Group Exmaple: polars
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
Output = fe.FE0_AutoRollStats(data=data, ArgsList=None, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
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
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
    
## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
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
# DiffDateVariables = 'CalendarDateColumn'
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
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
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
# DiffDateVariables = 'CalendarDateColumn'
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
Output = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
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
# DiffDateVariables = 'CalendarDateColumn'
# DiffGroupVariables = None
# NLag1 = 0
# NLag2 = 1
# Sort=True
# Processing = 'datatable'
# InputFrame = 'datatable'
# OutputFrame = 'datatable'

#########################################################################################################
#########################################################################################################

# QA FE0_AutoDiff
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe

# FE1_AutoCalendarVariables
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE1_AutoCalendarVariables(data=data, ArgsList=None, DateColumnNames = 'CalendarDateColumn', CalendarVariables = ['wday','mday','wom','month','quarter','year'], Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)

#########################################################################################################
#########################################################################################################

# Example: datatable
import timeit
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']


# Example: polars
import retrofit
from retrofit import FeatureEngineering as fe
import polars as pl
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
Output = fe.FE1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  Processing='polars', 
  InputFrame='polars', 
  OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
data = Output['data']
ArgsList = Output['ArgsList']

#########################################################################################################
#########################################################################################################

# FE2_AutoDataParition
import timeit
import datatable as dt
import polars as pl
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import utils as u

# datatable random Example
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
DataSets = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName='CalendarDateColumn', 
  PartitionType='random', 
  Ratios=[0.70,0.20,0.10], 
  Sort = False,
  ByVariables=None, 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
ArgsList = DataSets['ArgsList']

# data=data
# ArgsList=None
# DateColumnName='CalendarDateColumn'
# PartitionType='random'
# Ratios=[0.70,0.20,0.10]
# Sort = False
# ByVariables=None
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

# polars random Example
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
DataSets = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName='CalendarDateColumn', 
  PartitionType='random', 
  Ratios=[0.70,0.20,0.10], 
  ByVariables=None, 
  Sort = False,
  Processing='polars', 
  InputFrame='polars', 
  OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
ArgsList = DataSets['ArgsList']

# data=data
# ArgsList=None
# DateColumnName='CalendarDateColumn'
# PartitionType='random'
# Ratios=[0.70,0.20,0.10]
# Sort = False
# ByVariables=None
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'

# datatable time Example
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
DataSets = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName='CalendarDateColumn', 
  PartitionType='time', 
  Ratios=[0.70,0.20,0.10], 
  Sort = True,
  ByVariables=None, 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
ArgsList = DataSets['ArgsList']

# data=data
# ArgsList=None
# DateColumnName='CalendarDateColumn'
# PartitionType='time'
# Ratios=[0.70,0.20,0.10]
# Sort = False
# ByVariables=None
# Processing='datatable'
# InputFrame='datatable'
# OutputFrame='datatable'

# polars time Example
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
DataSets = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName='CalendarDateColumn', 
  PartitionType='time', 
  Ratios=[0.70,0.20,0.10], 
  ByVariables=None, 
  Sort = True,
  Processing='polars', 
  InputFrame='polars', 
  OutputFrame='polars')
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
ArgsList = DataSets['ArgsList']

# data=data
# ArgsList=None
# DateColumnName='CalendarDateColumn'
# PartitionType='time'
# Ratios=[0.70,0.20,0.10]
# Sort = False
# ByVariables=None
# Processing='polars'
# InputFrame='polars'
# OutputFrame='polars'

for i in data.shape[1]:
  if  not isinstance(data[i].dtype, pl.Categorical)
  data[i] = data[i].cast(pl.Categorical)

data.sort(DateColumnName, reverse = False, in_place = True)
