![Version: 0.1.7](https://img.shields.io/static/v1?label=Version&message=0.1.7&color=blue&?style=plastic)
![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![Build: Passing](https://img.shields.io/static/v1?label=Build&message=passing&color=brightgreen)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=default)](http://makeapullrequest.com)
[![GitHub Stars](https://img.shields.io/github/stars/AdrianAntico/RetroFit.svg?style=social)](https://github.com/AdrianAntico/retrofit)

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/PackageLogo.PNG' align='center' width='1000' />

Table of Contents
- [**Quick Note**](#quick-note)
- [**Installation**](#installation)
- [**Machine Learning Note**](#machine-learning-note)

Documentation + Code Examples
- [**Machine Learning**](#machine-learning)


## **Quick Note**
This package is currently in its beginning stages. I'll be working off a blueprint from my R package AutoQuant so there should be minimal breakages upon new releases, only non-breaking enhancements and additions. 

## **Installation**
```python
# Most up-to-date
pip install git+https://github.com/AdrianAntico/RetroFit.git#egg=retrofit

# From pypi
pip install retrofit==0.1.7

# Check out R package AutoQuant
https://github.com/AdrianAntico/AutoQuant
```


## **Machine Learning Note**

> Machine Learning Training: the goal here is enable the data scientist or machine learning engineer to effortlessly build any number of models with full optionality to tweak all available underlying parameters offered by the various algorithms. The underlying data can come from datatable or polars which means you'll be able to model with bigger data than if you were utilizing pandas. All models come with the ability to generate comprehensive evaluation metrics, evaluation plots, importances, and feature insights. Scoring should be seamless, from regenerating features for scoring to the actual scoring. The RetroFit class makes this super easy, fast, with minimal memory utilization.



<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/Documentation.PNG' align='center' width='1000' />


### **RetroFit Class**
<p>

<details><summary>CatBoost Examples</summary>
<p>

<details><summary>Regression Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets
from QuickEcharts import Charts
from retrofit import MachineLearning as ml


# Load some data
FilePath = f'{os.getcwd()}/RetroFit/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)


# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)


# Initialize RetroFit
model = ml.RetroFit(TargetType="regression", Algorithm="catboost")


# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Leads",
  NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'],
  CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3'],
  TextColumnNames=None,
  WeightColumnName=None,
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()


# Update algo args for GPU
model.update_model_parameters(
    task_type='GPU',
    sampling_frequency=None,
    rsm=1.0
)


# Train Model
model.train()


# Score train, validation, and test; store internally
model.score()


# Inspect scored data
model.ScoredData["train"]
model.ScoredData["validation"]
model.ScoredData["test"]


# Evaluate scored data
global_eval = model.evaluate(
    DataName="test"
)

# Per segment
segment_eval = model.evaluate(
    DataName="test",
    ByVariables="MarketingSegments"
)

# Per (Segment, Month)
segment_date_eval = model.evaluate(
    DataName="test",
    ByVariables=["MarketingSegments", "CalendarDateColumn"]
)
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python

```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python

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

```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python

```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python

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

```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python

```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python

```

</p>
</details>

</p>
</details>




</p>
</details>


</p>
</details>


## **Visualization**
<p>

<details><summary>Expand to view content</summary>
<p>

Code here

</p>
</details>
