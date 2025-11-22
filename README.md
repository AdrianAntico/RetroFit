![Version: 0.1.7](https://img.shields.io/static/v1?label=Version&message=0.1.7&color=blue&?style=plastic)
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
- [**Supervised Learning**](#supervised-learning)


## **Quick Note**
This package is currently in its beginning stages. Starting off will be supervised learning functions for regression, classification, and multiclass types. I incorporate usage of my companion packages:

- `PolarsFE`: feature engineering with polars
- `QuickEcharts`: visualization with Echarts.js

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

<details><summary>Expand to view content</summary>
<p>

> Machine Learning Training: the goal here is enable the data scientist or machine learning engineer to effortlessly build any number of models with full optionality to tweak all available underlying parameters offered by the various algorithms. The underlying data can come from pandas or polars (polars preferred) which means you'll be able to model with bigger data than if you were utilizing pandas. All models come with the ability to generate comprehensive evaluation metrics, evaluation plots, importances, and feature insights. The RetroFit class makes this super easy, fast, with minimal memory utilization.

</p>
</details>

<br>

# Code Examples


### **Supervised Learning**
<details><summary>Code</summary>
<p>



<p>

### CatBoost

<details><summary>Code Examples</summary>
<p>

<details><summary>Regression Training</summary>
<p>

```python
# Setup environment
import os
import polars as pl
from PolarsFE import datasets
from QuickEcharts import Charts
from retrofit import MachineLearning as ml
from retrofit import utils


# Load some data
df = utils.make_retrofit_demo_data(
    n_rows=50_000,
    n_segments=5,
    seed=42,
)

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
    rsm=1.0,
    iterations = 200
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

# Get variable importance
imp = model.compute_feature_importance()

# Get interaction importance
interact = model.compute_catboost_interaction_importance()

# Model Calibration Tables
cal = model.build_regression_calibration_table(
    DataName="test",
    binning="quantile"
)

# Store plot in working directory
model.plot_regression_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)

# Actual vs Predicted Scatterplot
model.plot_regression_scatter(
    DataName="test",
    SampleSize=15000,
    plot_name=f"{os.getcwd()}/my_scatter_plot",
    Theme="dark"
)
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
# Setup environment
import os
import polars as pl
from PolarsFE import datasets
from QuickEcharts import Charts
from retrofit import MachineLearning as ml
from retrofit import utils


# Load some data
df = utils.make_retrofit_demo_data(
    n_rows=50_000,
    n_segments=5,
    seed=42,
)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="classification", Algorithm="catboost")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label_binary",
  NumericColumnNames=['XREGS1','XREGS2','XREGS3'],
  CategoricalColumnNames=['MarketingSegments','MarketingSegments2','MarketingSegments3'],
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
    rsm=1.0,
    iterations=200
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

# Get variable importance
imp = model.compute_feature_importance()

# Get interaction importance
interact = model.compute_catboost_interaction_importance()

# Store plot in working directory
model.plot_classification_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)

# ROC Plot
model.plot_classification_roc(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_roc_plot",
)

# PR Plot
model.plot_classification_pr(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_pr_plot"
)
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
import os
import polars as pl
from PolarsFE import datasets
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="multiclass", Algorithm="catboost")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label",
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
    bootstrap_type='Bayesian',
    rsm=1.0,
    iterations=200,
    subsample=None,
    sampling_frequency=None
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

# Get variable importance
imp = model.compute_feature_importance()

# Get interaction importance
interact = model.compute_catboost_interaction_importance()
```

</p>
</details>


</p>
</details>


### XGBoost

<details><summary>Code Examples</summary>
<p>


<details><summary>Regression Training</summary>
<p>


```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="regression", Algorithm="xgboost")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Leads",
  NumericColumnNames=[
    'XREGS1',
    'XREGS2',
    'XREGS3',
    'MarketingSegments_TargetEncode',
    'MarketingSegments2_TargetEncode',
    'MarketingSegments3_TargetEncode'
  ],
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_boost_round=200,
    num_parallel_tree=4,
    max_depth=4
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
    ByVariables="MarketingSegments_TargetEncode"
)

# Per (Segment, Month)
segment_date_eval = model.evaluate(
    DataName="test",
    ByVariables=["MarketingSegments_TargetEncode", "CalendarDateColumn"]
)

# Get variable importance
imp = model.compute_feature_importance()

# Model Calibration Tables
cal = model.build_regression_calibration_table(
    DataName="test",
    binning="quantile"
)

# Store plot in working directory
model.plot_regression_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Turn Label into a binary target variable
df = df.with_columns(
    pl.when(pl.col("XREGS1") > 200).then(1)
      .otherwise(0)
      .alias("Label")
)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Label",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="classification", Algorithm="xgboost")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label",
  NumericColumnNames=[
    'XREGS1',
    'XREGS2',
    'XREGS3',
    'MarketingSegments_TargetEncode',
    'MarketingSegments2_TargetEncode',
    'MarketingSegments3_TargetEncode'
  ],
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_boost_round=200,
    num_parallel_tree=4,
    max_depth=4
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
    ByVariables="MarketingSegments_TargetEncode"
)

# Per (Segment, Month)
segment_date_eval = model.evaluate(
    DataName="test",
    ByVariables=["MarketingSegments_TargetEncode", "CalendarDateColumn"]
)

# Get variable importance
imp = model.compute_feature_importance()

# Store plot in working directory
model.plot_classification_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)

# ROC Plot
model.plot_classification_roc(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_roc_plot",
)

# PR Plot
model.plot_classification_pr(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_pr_plot"
)
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Label",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="multiclass", Algorithm="xgboost")

# Model Variables
drop_cols = ['CalendarDateColumn', 'Label']
features = [c for c in DataSets[2].columns if c not in drop_cols]

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label",
  NumericColumnNames=features,
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_boost_round=200,
    num_parallel_tree=4,
    max_depth=4
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

# Get variable importance
imp = model.compute_feature_importance()
```

</p>
</details>

</p>
</details>


### LightGBM

<details><summary>Code Examples</summary>
<p>


<details><summary>Regression Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="regression",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="regression", Algorithm="lightgbm")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Leads",
  NumericColumnNames=[
    'XREGS1',
    'XREGS2',
    'XREGS3',
    'MarketingSegments_TargetEncode',
    'MarketingSegments2_TargetEncode',
    'MarketingSegments3_TargetEncode'
  ],
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_iterations=200,
    max_depth=6,
    min_data_in_leaf=2
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
    ByVariables="MarketingSegments_TargetEncode"
)

# Per (Segment, Month)
segment_date_eval = model.evaluate(
    DataName="test",
    ByVariables=["MarketingSegments_TargetEncode", "CalendarDateColumn"]
)

# Get variable importance
imp = model.compute_feature_importance()

# Model Calibration Tables
cal = model.build_regression_calibration_table(
    DataName="test",
    binning="quantile"
)

# Store plot in working directory
model.plot_regression_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)
```

</p>
</details>


<details><summary>Classification Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Turn Label into a binary target variable
df = df.with_columns(
    pl.when(pl.col("XREGS1") > 200).then(1)
      .otherwise(0)
      .alias("Label")
)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="classification",
    group_variables=categorical_cols,
    target_variable="Label",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="classification", Algorithm="lightgbm")

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label",
  NumericColumnNames=[
    'XREGS1',
    'XREGS2',
    'XREGS3',
    'MarketingSegments_TargetEncode',
    'MarketingSegments2_TargetEncode',
    'MarketingSegments3_TargetEncode'
  ],
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_iterations=200,
    max_depth=6,
    min_data_in_leaf=2
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
    ByVariables="MarketingSegments_TargetEncode"
)

# Per (Segment, Month)
segment_date_eval = model.evaluate(
    DataName="test",
    ByVariables=["MarketingSegments_TargetEncode", "CalendarDateColumn"]
)

# Get variable importance
imp = model.compute_feature_importance()

# Store plot in working directory
model.plot_classification_calibration(
    DataName="test",
    n_bins=20,
    binning="quantile",
    plot_name=f"{os.getcwd()}/my_calibration_plot"
)

# ROC Plot
model.plot_classification_roc(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_roc_plot",
)

# PR Plot
model.plot_classification_pr(
    DataName="test",
    plot_name=f"{os.getcwd()}/my_pr_plot"
)
```

</p>
</details>


<details><summary>MultiClass Training</summary>
<p>

```python
# Setup Environment
import os
import polars as pl
from PolarsFE import datasets, character
from QuickEcharts import Charts
from retrofit import MachineLearning as ml

# Load some data
FilePath = f'{os.getcwd()}/retrofit/datasets/BenchmarkData.csv'
df = pl.read_csv(FilePath)

# Get TrainData, ValidationData, and TestData
DataSets = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.7, 0.2, 0.1]
)

# Create target encodings for categorical variables
categorical_cols = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3']
output = character.categorical_encoding(
    data=DataSets[0],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=False,
    keep_original_factors=False
)

# Collect data and encodings
DataSets[0] = output['data']
encodings = output['factor_components']

# Note parameter: scoring=True
DataSets[1] = character.categorical_encoding(
    data=DataSets[1],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Leads",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Note parameter: scoring=True
DataSets[2] = character.categorical_encoding(
    data=DataSets[2],
    ML_Type="multiclass",
    group_variables=categorical_cols,
    target_variable="Label",
    method="target_encoding",
    scoring=True,
    supply_factor_level_list=encodings,
    keep_original_factors=False
)

# Initialize RetroFit
model = ml.RetroFit(TargetType="multiclass", Algorithm="lightgbm")

# Model Variables
drop_cols = ['CalendarDateColumn', 'Label']
features = [c for c in DataSets[2].columns if c not in drop_cols]

# Create algo-specific model data
model.create_model_data(
  TrainData=DataSets[0],
  ValidationData=DataSets[1],
  TestData=DataSets[2],
  TargetColumnName="Label",
  NumericColumnNames=features,
  Threads=-1
)

# Print default parameter settings
model.print_algo_args()

# Update algo args
model.update_model_parameters(
    num_iterations=200,
    max_depth=6,
    min_data_in_leaf=2
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

# Get variable importance
imp = model.compute_feature_importance()
```

</p>
</details>

</p>
</details>




</p>
</details>


</p>
</details>




### **Model Evaluation Visuals**
<p>

<details><summary>Images</summary>
<p>



<details><summary>Calibration Plot</summary>
<p>

<br> 

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/Regression_Calibration_Plot.PNG' align='center' width='1000' />

</p>
</details>



<details><summary>Regression Scatter Plot</summary>
<p>

<br> 

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/Regression_Scatter_Plot.PNG' align='center' width='1000' />

</p>
</details>



<details><summary>ROC Plot</summary>
<p>

<br> 

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/ROC_Plot.PNG' align='center' width='1000' />

</p>
</details>



<details><summary>Precision-Recall Plot</summary>
<p>

<br> 

<img src='https://raw.githubusercontent.com/AdrianAntico/RetroFit/main/images/PR_Plot.PNG' align='center' width='1000' />

</p>
</details>





</p>
</details>


</p>
</details>
