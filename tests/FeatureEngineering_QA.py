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

# Run function
t_start = timeit.default_timer()
data1 = FE.FE0_AutoLags(
  data=data, 
  LagPeriods=1, 
  LagColumnNames='Leads', 
  DateColumnName='CalendarDateColumn', 
  ByVariables=None, 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args = False)
t_end = timeit.default_timer()
print(t_end - t_start)
del Output
print(data1.names)

# No Group Example: polars

# Instantiate Feature Engineering Class
FE = pfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = pl.read_csv(FilePath)
t_start = timeit.default_timer()
data2 = FE.FE0_AutoLags(
  data=data,
  LagPeriods=1,
  LagColumnNames='Leads',
  DateColumnName='CalendarDateColumn',
  ByVariables=None,
  ImputeValue=-1.0,
  Sort=True,
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data2.columns)

# Group Example, Single Lag: datatable

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
t_start = timeit.default_timer()
data1 = FE.FE0_AutoLags(
  data=data, 
  LagPeriods=1, 
  LagColumnNames='Leads', 
  DateColumnName='CalendarDateColumn', 
  ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data1.names)

# Group Exmaple: polars

# Instantiate Feature Engineering Class
FE = pfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = pl.read_csv(FilePath)
t_start = timeit.default_timer()
data2 = FE.FE0_AutoLags(
  data=data, 
  LagPeriods=1, 
  LagColumnNames='Leads', 
  DateColumnName='CalendarDateColumn', 
  ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], 
  ImputeValue=-1.0, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data2.columns)

# Group and Multiple Periods and LagColumnNames: datatable

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
t_start = timeit.default_timer()
data1 = FE.FE0_AutoLags(
  data=data, 
  LagPeriods=[1,3,5], 
  LagColumnNames=['Leads','XREGS1'], 
  DateColumnName='CalendarDateColumn', 
  ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data1.names)

# Group and Multiple Periods and LagColumnNames: datatable

# Instantiate Feature Engineering Class
FE = pfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = pl.read_csv(FilePath)
t_start = timeit.default_timer()
data2 = FE.FE0_AutoLags(
  data=data, 
  LagPeriods=[1,3,5],
  LagColumnNames=['Leads','XREGS1'], 
  DateColumnName='CalendarDateColumn', 
  ByVariables=['MarketingSegments','MarketingSegments2','MarketingSegments3', 'Label'], 
  ImputeValue=-1.0, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data2.columns)

#########################################################################################################
#########################################################################################################

# Test Function
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe

# Group Example:

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

t_start = timeit.default_timer()
data = FE.FE0_AutoRollStats(
  data=data, 
  RollColumnNames='Leads', 
  DateColumnName='CalendarDateColumn', 
  ByVariables=None, 
  MovingAvg_Periods=[3,5,7], 
  MovingSD_Periods=[3,5,7], 
  MovingMin_Periods=[3,5,7], 
  MovingMax_Periods=[3,5,7], 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Run function
t_start = timeit.default_timer()
data = FE.FE0_AutoRollStats(
  data=data, 
  RollColumnNames=['Leads','XREGS1'], 
  DateColumnName='CalendarDateColumn', 
  ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], 
  MovingAvg_Periods=[3,5,7], 
  MovingSD_Periods=[3,5,7], 
  MovingMin_Periods=[3,5,7], 
  MovingMax_Periods=[3,5,7], 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)

## No Group Example:
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Run function
t_start = timeit.default_timer()
data = FE.FE0_AutoRollStats(
  data=data, 
  RollColumnNames='Leads', 
  DateColumnName='CalendarDateColumn', 
  ByVariables=None, 
  MovingAvg_Periods=[3,5,7], 
  MovingSD_Periods=[3,5,7], 
  MovingMin_Periods=[3,5,7], 
  MovingMax_Periods=[3,5,7], 
  ImputeValue=-1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)

#########################################################################################################
#########################################################################################################

# Test Function
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe

## Group Example:
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

t_start = timeit.default_timer()
data = FE.FE0_AutoDiff(
  data=data, 
  DateColumnName = 'CalendarDateColumn', 
  ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], 
  DiffNumericVariables = 'Leads', 
  DiffDateVariables = 'CalendarDateColumn', 
  DiffGroupVariables = None, 
  NLag1 = 0, 
  NLag2 = 1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv')
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

t_start = timeit.default_timer()
data = FE.FE0_AutoDiff(
  data=data, 
  DateColumnName = 'CalendarDateColumn',
  ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], 
  DiffNumericVariables = 'Leads', 
  DiffDateVariables = 'CalendarDateColumn', 
  DiffGroupVariables = None, 
  NLag1 = 0, 
  NLag2 = 1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)

## No Group Example:
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

t_start = timeit.default_timer()
data = FE.FE0_AutoDiff(
  data=data, 
  DateColumnName = 'CalendarDateColumn', 
  ByVariables = None, 
  DiffNumericVariables = 'Leads', 
  DiffDateVariables = 'CalendarDateColumn', 
  DiffGroupVariables = None, 
  NLag1 = 0, 
  NLag2 = 1, 
  Sort=True, 
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
print(data.names)

#########################################################################################################
#########################################################################################################

# Test Function
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import PolarsFE as pfe

# DatatableFE
 
# Data can be created using the R package RemixAutoML and function FakeDataGenerator
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

t_start = timeit.default_timer()
data = FE.AutoCalendarVariables(
  data=data, 
  DateColumnNames = 'CalendarDateColumn',
  CalendarVariables = ['wday','mday','month','quarter','year'],
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
data.names

# PolarsFE

# Instantiate Feature Engineering Class
FE = pfe.FE()

t_start = timeit.default_timer()
data = FE.AutoCalendarVariables(
  data=data,
  DateColumnNames = 'CalendarDateColumn',
  CalendarVariables = ['wday','mday','month','quarter','year'],
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
data.names

#########################################################################################################
#########################################################################################################

# Example: datatable
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import PolarsFE as pfe

# DatatableFE

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Run function
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
t_start = timeit.default_timer()
data = FE.FE1_DummyVariables(
  data=data, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  use_saved_args=False)
t_end = timeit.default_timer()
t_end - t_start

# Example: polars
# DatatableFE

# Instantiate Feature Engineering Class
FE = pfe.FE()

# Run function
data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
t_start = timeit.default_timer()
data = FE.FE1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  use_saved_args=False)
t_end = timeit.default_timer()
t_end - t_start

#########################################################################################################
#########################################################################################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/RegressionData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Create some lags
data = FE.FE0_AutoLags(
    data,
    LagColumnNames=['Independent_Variable1', 'Independent_Variable2'],
    DateColumnName='DateTime',
    ByVariables='Factor_1',
    LagPeriods=[1,2],
    ImputeValue=-1,
    Sort=True,
    use_saved_args=False)

# Create some rolling stats
data = FE.FE0_AutoRollStats(
    data,
    RollColumnNames=['Independent_Variable1','Independent_Variable2'],
    DateColumnName='DateTime',
    ByVariables='Factor_1',
    MovingAvg_Periods=[1,2],
    MovingSD_Periods=[2,3],
    MovingMin_Periods=[1,2],
    MovingMax_Periods=[1,2],
    ImputeValue=-1,
    Sort=True,
    use_saved_args=False)

# Create some diffs
data = FE.FE0_AutoDiff(
    data,
    DateColumnName='DateTime',
    ByVariables=['Factor_1','Factor_2','Factor_3'],
    DiffNumericVariables='Independent_Variable1',
    DiffDateVariables=None,
    DiffGroupVariables=None,
    NLag1=0,
    NLag2=1,
    Sort=True,
    use_saved_args=False)

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    self,
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=False,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

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
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
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
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
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
