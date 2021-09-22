![Version: 0.1.7](https://img.shields.io/static/v1?label=Version&message=0.1.7&color=blue&?style=plastic)
![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![Build: Passing](https://img.shields.io/static/v1?label=Build&message=passing&color=brightgreen)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=default)](http://makeapullrequest.com)
[![GitHub Stars](https://img.shields.io/github/stars/AdrianAntico/RetroFit.svg?style=social)](https://github.com/AdrianAntico/retrofit)

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/PackageLogo.PNG' align='center' width='1000' />

## Quick Note
This package is currently in its beginning stages. I'll be working off a blueprint from my R package RemixAutoML so there should be minimal breakages upon new releases, only non-breaking enhancements and additions. 

## Installation
```
# Most up-to-date
pip install git+https://github.com/AdrianAntico/RetroFit.git#egg=retrofit

# From pypi
pip install retrofit==0.1.7

# Check out R package RemixAutoML
https://github.com/AdrianAntico/RemixAutoML
```


## Feature Engineering

> Feature Engineering - Some of the feature engineering functions can only be found in this package. I believe feature engineering is your best bet for improving model performance. I have functions that cover all feature types. There are feature engineering functions for numeric data, categorical data, text data, and date data. They are all designed to generate features for training and scoring pipelines and they run extremely fast with low memory utilization. The Feature Engineering class offers the user the ability to have features generated using datatable, polars, or pandas for all feature engineering and data wrangling related methods. All methods collect paramter settings which will be used for scoring pipelines without the need for the user to save them. This makes life really easy when designing training and scoring pipelines. 

## Machine Learning

> Machine Learning Training: the goal here is enable the data scientist or machine learning engineer to effortlessly build any number of models with full optionality to tweak all available underlying parameters offered by the various algorithms. The underlying data can come from datatable or polars which means you'll be able to model with bigger data than if you were utilizing pandas. All models come with the ability to generate comprehensive evaluation metrics, evaluation plots, importances, and feature insights. Scoring should be seamless, from regenerating features for scoring to the actual scoring. The RetroFit class makes this super easy, fast, with minimal memory utilization.



<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/Documentation.PNG' align='center' width='1000' />




## Feature Engineering
<p>

<details><summary>Expand to view content</summary>
<p>


### FE0 Feature Engineering: Row-Dependence
<p>

<details><summary>Expand to view content</summary>
<p>


#### **FE0_AutoLags()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE0_AutoLags()</code> Automatically generate any number of lags, for any number of columns, by any number of By-Variables, using datatable.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
```

</p>
</details>



#### **FE0_AutoRollStats()**
<p>


<details><summary>Function Description</summary>
<p>
 
<code>FE0_AutoRollStats()</code> Automatically generate any number of moving averages, moving standard deviations, moving mins and moving maxs from any number of source columns, by any number of By-Variables, using datatable.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
```

</p>
</details>



#### **FE0_AutoDiff()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE0_AutoDiff()</code> Automatically generate any number of differences from any number of source columns, for numeric, character, and date columns, by any number of By-Variables, using datatable.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
```

</p>
</details>



</p>
</details>


### FE1 Feature Engineering: Row-Independence
<p>

<details><summary>Expand to view content</summary>
<p>

#### **FE1_AutoCalendarVariables()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE1_AutoCalendarVariables()</code> Automatically generate calendar variables from your datatable.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
```

</p>
</details>





#### **FE1_DummyVariables()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE1_DummyVariables()</code> Automatically generate dummy variables for user supplied categorical columns

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
data = pl.read_csv('C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv')
t_start = timeit.default_timer()
data = FE.FE1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  use_saved_args=False)
t_end = timeit.default_timer()
t_end - t_start
```

</p>
</details>




</p>
</details>



### FE2 Feature Engineering: Full-Data-Set
<p>

<details><summary>Expand to view content</summary>
<p>

#### **FE2_ColTypeConversions()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE2_ColTypeConversions()</code> Automatically convert column types required by certain models

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
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
```

</p>
</details>



#### **FE2_AutoDataParition()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>FE2_AutoDataParition()</code> Automatically create data sets for training based on random or time based splits

</p>
</details>

<details><summary>Code Example</summary>
<p>


```python
# FE2_AutoDataParition Example

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import PolarsFE as pfe

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/RegressionData.csv')
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# datatable random Example
t_start = timeit.default_timer()
DataSets = FE.FE2_AutoDataParition(
  data=data,
  DateColumnName='CalendarDateColumn',
  PartitionType='random',
  Ratios=[0.70,0.20,0.10],
  Sort = False,
  ByVariables=None,
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']

# polars random Example
data = pl.read_csv(FilePath)
t_start = timeit.default_timer()
DataSets = FE.FE2_AutoDataParition(
  data=data,
  DateColumnName='CalendarDateColumn',
  PartitionType='random',
  Ratios=[0.70,0.20,0.10],
  ByVariables=None,
  Sort = False,
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']

# datatable time Example
data = dt.fread(FilePath)
t_start = timeit.default_timer()
DataSets = FE.FE2_AutoDataParition(
  data=data,
  DateColumnName='CalendarDateColumn',
  PartitionType='time',
  Ratios=[0.70,0.20,0.10],
  Sort = True,
  ByVariables=None,
  use_saved_args=False)
t_end = timeit.default_timer()
print(t_end - t_start)
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']

# polars time Example
data = pl.read_csv(FilePath)
t_start = timeit.default_timer()
DataSets = FE.FE2_AutoDataParition(
  data=data,
  DateColumnName='CalendarDateColumn',
  PartitionType='time',
  Ratios=[0.70,0.20,0.10],
  ByVariables=None,
  Sort = True,
  use_saved_args=False)
t_end = timeit.default_timer()
t_end - t_start
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
```

</p>
</details>




</p>
</details>


### FE3 Feature Engineering: Model-Based
<p>

<details><summary>Expand to view content</summary>
<p>

##### Coming soon

</p>
</details>

</p>
</details>



## Machine Learning
<p>
 
<details><summary>Expand to view content</summary>
<p>


### ML0 Machine Learning: Prepare for Modeling
<p>

<details><summary>Expand to view content</summary>
<p>


#### **ML0_Parameters()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>ML0_Parameters()</code> Automatically generate parameters for modeling. User can update the parameters as desired.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
# Setup Environment
import pkg_resources
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Create partitioned data sets
Data = fe.FE2_AutoDataParition(
  data=data, 
  ArgsList=None, 
  DateColumnName=None, 
  PartitionType='random', 
  Ratios=[0.7,0.2,0.1], 
  ByVariables=None, 
  Sort=False, 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')

# Prepare modeling data sets
DataSets = ml.ML0_GetModelData(
  Processing='catboost',
  TrainData=Data['TrainData'],
  ValidationData=Data['ValidationData'],
  TestData=Data['TestData'],
  ArgsList=None,
  TargetColumnName='Leads',
  NumericColumnNames=['XREGS1','XREGS2','XREGS3'],
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'],
  TextColumnNames=None,
  WeightColumnName=None,
  Threads=-1,
  InputFrame='datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms='CatBoost', 
  TargetType='Regression', 
  TrainMethod='Train')
```

</p>
</details>



#### **ML0_GetModelData()**
<p>

<details><summary>Function Description</summary>
<p>
 
<code>ML0_GetModelData()</code> Automatically create data sets chosen ML algorithm. Currently supports catboost, xgboost, and lightgbm.

</p>
</details>

<details><summary>Code Example</summary>
<p>

```python
# ML0_GetModelData Example:
import pkg_resources
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

############################################################################################
# CatBoost
############################################################################################

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
    
# Create partitioned data sets
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

# Collect partitioned data
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
del DataSets

# Create catboost data sets
DataSets = ml.ML0_GetModelData(
  TrainData=TrainData, 
  ValidationData=ValidationData, 
  TestData=TestData, 
  ArgsList=None, 
  TargetColumnName='Leads', 
  NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'], 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'], 
  TextColumnNames=None, 
  WeightColumnName=None, 
  Threads=-1, 
  Processing='catboost', 
  InputFrame='datatable')
  
# Collect catboost training data
catboost_train = DataSets['train_data']
catboost_validation = DataSets['validation_data']
catboost_test = DataSets['test_data']

############################################################################################
# XGBoost
############################################################################################

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
    
# Create partitioned data sets
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

# Collect partitioned data
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
del DataSets

# Create xgboost data sets
DataSets = ml.ML0_GetModelData(
  TrainData=TrainData, 
  ValidationData=ValidationData, 
  TestData=TestData, 
  ArgsList=None, 
  TargetColumnName='Leads', 
  NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'], 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'], 
  TextColumnNames=None, 
  WeightColumnName=None, 
  Threads=-1, 
  Processing='xgboost', 
  InputFrame='datatable')
  
# Collect xgboost training data
xgboost_train = DataSets['train_data']
xgboost_validation = DataSets['validation_data']
xgboost_test = DataSets['test_data']

############################################################################################
# LightGBM
############################################################################################

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)
    
# Create partitioned data sets
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

# Collect partitioned data
TrainData = DataSets['TrainData']
ValidationData = DataSets['ValidationData']
TestData = DataSets['TestData']
del DataSets

# Create lightgbm data sets
DataSets = ml.ML0_GetModelData(
  TrainData=TrainData, 
  ValidationData=ValidationData, 
  TestData=TestData, 
  ArgsList=None, 
  TargetColumnName='Leads', 
  NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'], 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'], 
  TextColumnNames=None, 
  WeightColumnName=None, 
  Threads=-1, 
  Processing='lightgbm', 
  InputFrame='datatable')
  
# Collect lightgbm training data
lightgbm_train = DataSets['train_data']
lightgbm_validation = DataSets['validation_data']
lightgbm_test = DataSets['test_data']
```

</p>
</details>


</p>
</details>



### **ML1 Machine Learning: RetroFit Class**
<p>

<details><summary>Expand to view content</summary>
<p>


<details><summary>Class Meta Information</summary>
<p>


<details><summary>Class Goals</summary>
<p>

```python
####################################
# Goals
####################################

Class Initialization
Model Initialization
Training
Feature Tuning
Grid Tuning
Model Scoring
Model Evaluation
Model Interpretation
```

</p>
</details>

<details><summary>Class Functions</summary>
<p>

```python
####################################
# Functions
####################################

ML1_Single_Train()
ML1_Single_Score()
ML1_Single_Evaluate()
PrintAlgoArgs()
```

</p>
</details>


<details><summary>Class Attributes</summary>
<p>

```python
####################################
# Attributes
####################################

self.ModelArgs = ModelArgs
self.ModelArgsNames = [*self.ModelArgs]
self.Runs = len(self.ModelArgs)
self.DataSets = DataSets
self.DataSetsNames = [*self.DataSets]
self.ModelList = dict()
self.ModelListNames = []
self.FitList = dict()
self.FitListNames = []
self.EvaluationList = dict()
self.EvaluationListNames = []
self.InterpretationList = dict()
self.InterpretationListNames = []
self.CompareModelsList = dict()
self.CompareModelsListNames = []
```

</p>
</details>

</p>
</details>



<details><summary>Ftrl Examples</summary>
<p>


<details><summary>Regression Training</summary>
<p>

```python
####################################
# Ftrl Regression
####################################

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
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'Ftrl',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'Ftrl', 
  TargetType = 'Regression', 
  TrainMethod = 'Train')

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'Ftrl')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2], 
  ModelName = x.ModelListNames[0], 
  Algorithm = 'Ftrl', 
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('Ftrl')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=None)

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_Ftrl_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo='Ftrl')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
####################################
# Ftrl Classification
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/ClassificationData.csv') 
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
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'Ftrl',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'Ftrl', 
  TargetType = 'Classification', 
  TrainMethod = 'Train')

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'Ftrl')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'Ftrl',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('Ftrl')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=1))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_Ftrl_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo='Ftrl')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
####################################
# Ftrl MultiClass
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/MultiClassData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'Ftrl',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'Ftrl',
  TargetType = 'MultiClass',
  TrainMethod = 'Train')

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'Ftrl')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'Ftrl',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('Ftrl')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=1))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_Ftrl_1').names

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo='Ftrl')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>

</p>
</details>


<details><summary>CatBoost Examples</summary>
<p>

<details><summary>Regression Training</summary>
<p>

```python
####################################
# CatBoost Regression
####################################

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
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'catboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'CatBoost', 
  TargetType = 'Regression', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('CatBoost').get('AlgoArgs')['iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'CatBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2], 
  ModelName = x.ModelListNames[0],
  Algorithm = 'CatBoost',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('CatBoost')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=None)

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_CatBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'CatBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
####################################
# CatBoost Classification
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/ClassificationData.csv') 
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
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'catboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'CatBoost', 
  TargetType = 'Classification', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('CatBoost').get('AlgoArgs')['iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'CatBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2], 
  ModelName = x.ModelListNames[0],
  Algorithm = 'CatBoost',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('CatBoost')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=0))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_CatBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'CatBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
####################################
# CatBoost MultiClass
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/MultiClassData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'catboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'CatBoost',
  TargetType = 'MultiClass',
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('CatBoost').get('AlgoArgs')['iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'CatBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2], 
  ModelName = x.ModelListNames[0],
  Algorithm = 'CatBoost',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('CatBoost')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=0))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_CatBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'CatBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


</p>
</details>



<details><summary>XGBoost Examples</summary>
<p>


<details><summary>Regression Training</summary>
<p>


```python
####################################
# XGBoost Regression
####################################

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

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_1','Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_1','Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
  data,
  DateColumnNames='DateTime',
  CalendarVariables=['wday','month','quarter'],
  use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
  data,
  Int2Float=True,
  Bool2Float=True,
  RemoveDateCols=True,
  RemoveStrCols=False,
  SkipCols=None,
  use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'xgboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'XGBoost', 
  TargetType = "Regression", 
  TrainMethod = "Train")

# Update iterations to run quickly
ModelArgs['XGBoost']['AlgoArgs']['num_boost_round'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'XGBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'XGBoost',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('XGBoost')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=None)

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_XGBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'XGBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
####################################
# XGBoost Classification
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/ClassificationData.csv') 
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

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_1','Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_1','Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'xgboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'XGBoost', 
  TargetType = 'Classification', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('XGBoost').get('AlgoArgs')['num_boost_round'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'XGBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'XGBoost',
  NewData = None)

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('XGBoost')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=0))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_XGBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'XGBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
####################################
# XGBoost MultiClass
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/MultiClassData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'xgboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'XGBoost',
  TargetType = 'MultiClass',
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('XGBoost').get('AlgoArgs')['num_boost_round'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'XGBoost')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'XGBoost',
  NewData = None)

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_XGBoost_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'XGBoost')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>

</p>
</details>


<details><summary>LightGBM Examples</summary>
<p>


<details><summary>Regression Training</summary>
<p>

```python
####################################
# LightGBM Regression
####################################

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

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_1','Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_1','Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'lightgbm',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'LightGBM', 
  TargetType = 'Regression', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('LightGBM').get('AlgoArgs')['num_iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'LightGBM')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'LightGBM')

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_LightGBM_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'LightGBM')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
####################################
# LightGBM Classification
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/ClassificationData.csv') 
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

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_1','Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_1','Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'lightgbm',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'LightGBM', 
  TargetType = 'Classification', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('LightGBM').get('AlgoArgs')['num_iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'LightGBM')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'LightGBM')

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('LightGBM')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=0))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_LightGBM_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'LightGBM')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
####################################
# LightGBM MultiClass
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
import retrofit
from retrofit import DatatableFE as dtfe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/MultiClassData.csv') 
data = dt.fread(FilePath)

# Instantiate Feature Engineering Class
FE = dtfe.FE()

# Dummify
data = FE.FE1_DummyVariables(
  data = data, 
  CategoricalColumnNames = ['Factor_2','Factor_3'],
  use_saved_args=False)
data = data[:, [name not in ['Factor_2','Factor_3'] for name in data.names]]

# Create Calendar Vars
data = FE.FE1_AutoCalendarVariables(
    data,
    DateColumnNames='DateTime',
    CalendarVariables=['wday','month','quarter'],
    use_saved_args=False)

# Type conversions for modeling
data = FE.FE2_ColTypeConversions(
    data,
    Int2Float=True,
    Bool2Float=True,
    RemoveDateCols=True,
    RemoveStrCols=False,
    SkipCols=None,
    use_saved_args=False)

# Drop Text Cols (no word2vec yet)
data = data[:, [z for z in data.names if z not in ['Comment']]]

# Create partitioned data sets
DataFrames = FE.FE2_AutoDataPartition(
  data, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False,
  use_saved_args = False)

# Features
Features = [z for z in list(data.names) if not z in ['Adrian','DateTime','Comment','Weights']]

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'lightgbm',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'LightGBM', 
  TargetType = 'MultiClass', 
  TrainMethod = 'Train')

# Update iterations to run quickly
ModelArgs.get('LightGBM').get('AlgoArgs')['num_iterations'] = 50

# Initialize RetroFit
x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

# Train Model
x.ML1_Single_Train(Algorithm = 'LightGBM')

# Score data
x.ML1_Single_Score(
  DataName = x.DataSetsNames[2],
  ModelName = x.ModelListNames[0],
  Algorithm = 'LightGBM')

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('LightGBM')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=dict(tpcost=0, fpcost=1, fncost=1, tncost=0))

# Metrics
metrics.keys()

# Scoring data names
x.DataSetsNames

# Scoring data
x.DataSets.get('Scored_test_data_LightGBM_1')

# Check ModelArgs Dict
x.PrintAlgoArgs(Algo = 'LightGBM')

# List of model names
x.ModelListNames

# List of model fitted names
x.FitListNames
```

</p>
</details>

</p>
</details>




</p>
</details>


</p>
</details>
