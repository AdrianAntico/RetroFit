![Version: 0.0.1](https://img.shields.io/static/v1?label=Version&message=0.0.1&color=blue&?style=plastic)
![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![Build: Passing](https://img.shields.io/static/v1?label=Build&message=passing&color=brightgreen)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=default)](http://makeapullrequest.com)
[![GitHub Stars](https://img.shields.io/github/stars/AdrianAntico/retrofit.svg?style=social)](https://github.com/AdrianAntico/retrofit)

<img src="https://github.com/AdrianAntico/retrofit/blob/main/images/PackageLogo.PNG" align="center" width="1000" />


## Installation
```
pip install git+https://github.com/AdrianAntico/RetroFit.git#egg=retrofit
```


## Feature Engineering

> Feature Engineering - Some of the feature engineering functions can only be found in this package. I believe feature engineering is your best bet for improving model performance. I have functions that cover all feature types. There are feature engineering functions for numeric data, categorical data, text data, and date data. They are all designed to generate features for training and scoring pipelines and they run extremely fast with low memory utilization. The package takes advantage of datatable for all feature engineering and data wrangling related functions which means you'll only have to go to big data tools if absolutely necessary.

## Machine Learning

> Machine Learning Training -

> Machine Learning Scoring -

> Machine Learning Evaluation -

> Machine Learning Interpretation -

## retrofit and RemixAutoML Blogs

<details><summary>Expand to view content</summary>
<p>

### Python retrofit and R RemixAutoML Blogs

[Sales Funnel Forecasting with ML using RemixAutoML](https://adrianantico.medium.com/sales-funnel-forecasting-using-ml-with-remixautoml-86361ce281b3)
 
[The Most Feature Rich ML Forecasting Methods Available](https://adrianantico.medium.com/the-most-feature-rich-ml-forecasting-methods-available-compliments-of-remixautoml-61b53daf42e6)

[AutoML Frameworks in R & Python](https://iamnagdev.com/2020/04/01/automl-frameworks-in-r-python/)

[AI for Small to Medium Size Businesses: A Management Take On The Challenges...](https://www.remixinstitute.com/blog/business-ai-for-small-to-medium-sized-businesses-with-remixautoml/#.XX-lD2ZlD8A)

[Why Machine Learning is more Practical than Econometrics in the Real World](https://medium.com/@adrianantico/machine-learning-vs-econometrics-in-the-real-world-4058095b1013)

[Build Thousands of Automated Demand Forecasts in 15 Minutes Using AutoCatBoostCARMA in R](https://www.remixinstitute.com/blog/automated-demand-forecasts-using-autocatboostcarma-in-r/#.XUIO1ntlCDM)

[Automate Your KPI Forecasts With Only 1 Line of R Code Using AutoTS](https://www.remixinstitute.com/blog/automate-your-kpi-forecasts-with-only-1-line-of-r-code-using-autots/#.XUIOr3tlCDM)

</p>
</details>

<img src="https://github.com/AdrianAntico/RemixAutoML/blob/master/Images/Documentation.PNG" align="center" width="1000" />




## Feature Engineering
<p>

<img src="https://github.com/AdrianAntico/RetroFit/blob/main/images/FeatureEngineeringAll.PNG" align="center" width="600" />

<details><summary>Expand to view feature engineering functions</summary>
<p>
 
### Feature Engineering: Date Feature Engineering

<details><summary>Expand to view content</summary>
<p>

#### Coming Soon

</p>
</details>

### Feature Engineering: Numeric Feature Engineering

<details><summary>Expand to view content</summary>
<p>

#### Coming Soon

</p>
</details>

### Feature Engineering: Categorical Feature Engineering

<details><summary>Expand to view content</summary>
<p>

#### Coming Soon

</p>
</details>

### Feature Engineering: Cross-Row Operations

<details><summary>Expand to view content</summary>
<p>

#### **AutoLags()**

<details><summary>Code Example</summary>
<p>

```
# Test Function
import datatable
import retrofit
from retrofit import TimeSeriesFeatures as ts
 
# Data can be created using the R package RemixAutoML and function FakeDataGenerator
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
## Group Example:
data = ts.AutoLags(data=data, LagPeriods=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
print(data.names)
    
## Group and Multiple Periods and LagColumnNames:
data = ts.AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = ts.AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
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
import datatable
import retrofit
from retrofit import TimeSeriesFeatures as ts

## Group Example:
import datatable as dt
from datatable import sort, f, by
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
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
import datatable
import retrofit
from retrofit import TimeSeriesFeatures as ts

## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)
    
## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = ts.AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
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


### Feature Engineering: Data Set Feature Engineering

<details><summary>Expand to view content</summary>
<p>

#### Coming Soon

</p>
</details>

### Feature Engineering: Model-Based Feature Engineering

<details><summary>Expand to view content</summary>
<p>

#### Coming Soon

</p>
</details>

</p>
</details>




## Machine Learning Training
<p>
 
<details><summary>Expand to view machine learning functions</summary>
<p>

#### Coming Soon

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
