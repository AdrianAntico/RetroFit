![Version: 0.0.4](https://img.shields.io/static/v1?label=Version&message=0.0.4&color=blue&?style=plastic)
![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![Build: Passing](https://img.shields.io/static/v1?label=Build&message=passing&color=brightgreen)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=default)](http://makeapullrequest.com)
[![GitHub Stars](https://img.shields.io/github/stars/AdrianAntico/RetroFit.svg?style=social)](https://github.com/AdrianAntico/retrofit)

<img src="https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/PackageLogo.PNG" align="center" width="1000" />

## Quick Note
This package is currently in its beginning stages. I'll be working off a blueprint from my R package RemixAutoML so there should be minimal breakages upon new releases, only non-breaking enhancements and additions. 

## Installation
```
# Most up-to-date
pip install git+https://github.com/AdrianAntico/RetroFit.git#egg=retrofit

# From pypi
pip install retrofit==0.0.4

# Check out R package RemixAutoML
https://github.com/AdrianAntico/RemixAutoML
```


## Feature Engineering

> Feature Engineering - Some of the feature engineering functions can only be found in this package. I believe feature engineering is your best bet for improving model performance. I have functions that cover all feature types. There are feature engineering functions for numeric data, categorical data, text data, and date data. They are all designed to generate features for training and scoring pipelines and they run extremely fast with low memory utilization. The package takes advantage of datatable or polars (user chooses) for all feature engineering and data wrangling related functions which means you'll only have to go to big data tools if absolutely necessary.

## Machine Learning

> Machine Learning Training -

> Machine Learning Scoring -

> Machine Learning Evaluation -

> Machine Learning Interpretation -



<img src="https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/Documentation.PNG" align="center" width="1000" />




## Feature Engineering

<details><summary>Expand to view content</summary>
<p>


### S0 Feature Engineering: Row-Dependence

<details><summary>Expand to view content</summary>
<p>


#### **AutoLags()**

<details><summary>Code Example</summary>
<p>

```
# Test Function
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
 
# Data can be created using the R package RemixAutoML and function FakeDataGenerator
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
## Group Example:
data = fe.S0_AutoLags(data=data, LagPeriods=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
print(data.names)
    
## Group and Multiple Periods and LagColumnNames:
data = fe.S0_AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = fe.S0_AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
print(data.names)
```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoLags()</code> Automatically generate any number of lags, for any number of columns, by any number of By-Variables, using datatable.

</p>
</details>

#### **AutoRollStats()**

<details><summary>Code Example</summary>
<p>

```
# Test Function
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe

## Group Example:
import datatable as dt
from datatable import sort, f, by
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)
```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoRollStats()</code> Automatically generate any number of moving averages, moving standard deviations, moving mins and moving maxs from any number of source columns, by any number of By-Variables, using datatable.

</p>
</details>


 
#### **AutoDiff()**

<details><summary>Code Example</summary>
<p>

```
# Test Function
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe

## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.S0_AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)
```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoDiff()</code> Automatically generate any number of differences from any number of source columns, for numeric, character, and date columns, by any number of By-Variables, using datatable.

</p>
</details>

</p>
</details>


### S1 Feature Engineering: Row-Independence

<details><summary>Expand to view content</summary>
<p>

#### **AutoCalendarVariables()**

<details><summary>Code Example</summary>
<p>

```
# Test Function
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
 
# Data can be created using the R package RemixAutoML and function FakeDataGenerator
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = fe.AutoCalendarVariables(
  data=data, 
  ArgsList=None, 
  DateColumnNames = 'CalendarDateColumn', 
  CalendarVariables = ['wday','mday','wom','month','quarter','year'], 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')

# Check
data.names
```

</p>
</details>


<details><summary>Function Description</summary>
<p>
 
<code>AutoCalendarVariables()</code> Automatically generate calendar variables from your datatable.

</p>
</details>


#### **S1_DummyVariables()**

<details><summary>Code Example</summary>
<p>

```
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
Output = fe.S1_DummyVariables(
  data=data, 
  ArgsList=None, 
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], 
  Processing='datatable', 
  InputFrame='datatable', 
  OutputFrame='datatable')
data = Output['data']
ArgsList = Output['ArgsList']
```

</p>
</details>


<details><summary>Function Description</summary>
<p>
 
<code>S1_DummyVariables()</code> Automatically generate dummy variables for user supplied categorical columns

</p>
</details>


</p>
</details>



### S2 Feature Engineering: Full-Data-Set

<details><summary>Expand to view content</summary>
<p>


#### **S2_AutoDataParition()**

<details><summary>Code Example</summary>
<p>


```
# Example
import datatable as dt
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import utils as u
    
# random
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
DataSets = fe.S2_AutoDataParition(
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
```

</p>
</details>


<details><summary>Function Description</summary>
<p>
 
<code>S2_AutoDataParition()</code> Automatically create data sets for training based on random or time based splits

</p>
</details>

</p>
</details>


### S3 Feature Engineering: Model-Based

<details><summary>Expand to view content</summary>
<p>

##### Coming soon

</p>
</details>

</p>
</details>



## Machine Learning Training
<p>
 
<details><summary>Expand to view content</summary>
<p>

### **ML0_GetModelData()**

<details><summary>Code Example</summary>
<p>

```
# ML0_GetModelData Example:
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# Load some data
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
# Create partitioned data sets
DataSets = fe.S2_AutoDataParition(
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
```

</p>
</details>


<details><summary>Function Description</summary>
<p>
 
<code>ML0_GetModelData()</code> Automatically create data sets chosen ML algorithm

</p>
</details>


</p>
</details>




## Machine Learning Scoring
<p>
 
<details><summary>Expand to view machine learning scoring functions</summary>
<p>

#### Coming Soon

</p>
</details>




## Machine Learning Evaluation
<p>
 
<details><summary>Expand to view machine learning evaluation functions</summary>
<p>

#### Coming Soon

</p>
</details>




## Machine Learning Interpretation
<p>
 
<details><summary>Expand to view machine learning interpretation functions</summary>
<p>

#### Coming Soon

</p>
</details>
