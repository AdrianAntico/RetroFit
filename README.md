![Version: 0.0.1](https://img.shields.io/static/v1?label=Version&message=0.0.1&color=blue&?style=plastic)
![Build: Passing](https://img.shields.io/static/v1?label=Build&message=passing&color=brightgreen)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=default)](http://makeapullrequest.com)
[![GitHub Stars](https://img.shields.io/github/stars/AdrianAntico/RemixAutoML.svg?style=social)](https://github.com/AdrianAntico/RetroFit.FeatureEngineering)

<img src="https://github.com/AdrianAntico/RetroFit.FeatureEngineering/blob/master/images/PackageLogo.png" align="center" width="1000" />

## Background

<details><summary>Expand to view content</summary>
<p>
 
> Automated Feature Engineering - 

> Feature Engineering - Some of the feature engineering functions can only be found in this package. I believe feature engineering is your best bet for improving model performance. I have functions that cover all feature types except image data. There are feature engineering functions for numeric data, categorical data, text data, and date data. They are all designed to generate features for training and scoring pipelines and they run extremely fast with low memory utilization. The package takes advantage of data.table for all feature engineering and data wrangling related functions which means I only have to go to big data tools if absolutely necessary.

> Documentation - Each exported function in the package has a help file and can be viewed in your RStudio session, e.g. <code>?RemixAutoML::ModelDataPrep</code>. Many of them come with examples coded up in the help files (at the bottom) that you can run to get a feel for how to set the parameters. There's also a listing of exported functions by category with code examples at the bottom of this readme. You can also jump into the R folder here to dig into the source code. 

</p>
</details>

## Installation

The Description File is designed to require only the minimum number of packages to install RemixAutoML. However, in order to utilize most of the functions in the package, you'll have to install additional libraries. I set it up this way on purpose. You don't need to install every single possible dependency if you are only interested in using a few of the functions. For example, if you only want to use CatBoost then intall the catboost package and forget about the h2o, xgboost, and lightgbm packages. This is one of the primary benefits of not hosting an R package on cran, as they require dependencies to be part of the Imports section on the Description File, which subsequently requires users to have all dependencies installed in order to install the package.

The minimal set of packages that need to be installed are below. The full list can be found by expanding the section (Expand to view content).
* polars
* 
* 
* 
* 
* 
* 

## RetroFit and RemixAutoML Blogs

<details><summary>Expand to view content</summary>
<p>

### Python RetroFit Blogs

### R RemixAutoML Blogs
[Sales Funnel Forecasting with ML using RemixAutoML](https://adrianantico.medium.com/sales-funnel-forecasting-using-ml-with-remixautoml-86361ce281b3)
 
[The Most Feature Rich ML Forecasting Methods Available](https://adrianantico.medium.com/the-most-feature-rich-ml-forecasting-methods-available-compliments-of-remixautoml-61b53daf42e6)

[AutoML Frameworks in R & Python](https://iamnagdev.com/2020/04/01/automl-frameworks-in-r-python/)

[AI for Small to Medium Size Businesses: A Management Take On The Challenges...](https://www.remixinstitute.com/blog/business-ai-for-small-to-medium-sized-businesses-with-remixautoml/#.XX-lD2ZlD8A)

[Why Machine Learning is more Practical than Econometrics in the Real World](https://medium.com/@adrianantico/machine-learning-vs-econometrics-in-the-real-world-4058095b1013)

[Build Thousands of Automated Demand Forecasts in 15 Minutes Using AutoCatBoostCARMA in R](https://www.remixinstitute.com/blog/automated-demand-forecasts-using-autocatboostcarma-in-r/#.XUIO1ntlCDM)

[Automate Your KPI Forecasts With Only 1 Line of R Code Using AutoTS](https://www.remixinstitute.com/blog/automate-your-kpi-forecasts-with-only-1-line-of-r-code-using-autots/#.XUIOr3tlCDM)

</p>
</details>

<img src="https://github.com/AdrianAntico/RemixAutoML/blob/master/Images/Documentation.PNG" align="center" width="725" />

## Feature Engineering <img src="https://github.com/AdrianAntico/RemixAutoML/blob/master/Images/FeatureEngineeringMenu.PNG" align="right" width="80" />

<details><summary>Expand to view content</summary>
<p>

<img src="https://github.com/AdrianAntico/RemixAutoML/blob/master/Images/FeatureEngineeringMenu.PNG" align="center" width="800" />

#### **AutoLagRollStats()** and **AutoLagRollStatsScoring()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoLagRollStats()</code> builds lags and rolling statistics by grouping variables and their interactions along with multiple different time aggregations if selected. Rolling stats include mean, sd, skewness, kurtosis, and the 5th - 95th percentiles. This function was inspired by the distributed lag modeling framework but I wanted to use it for time series analysis as well and really generalize it as much as possible. The beauty of this function is inspired by analyzing whether a baseball player will get a basehit or more in his next at bat. One easy way to get a better idea of the likelihood is to look at his batting average and his career batting average. However, players go into hot streaks and slumps. How do we account for that? Well, in comes the functions here. You look at the batting average over the last N to N+x at bats, for various N and x. I keep going though - I want the same windows for calculating the players standard deviation, skewness, kurtosis, and various quantiles over those time windows. I also want to look at all those measure but by using weekly data - as in, over the last N weeks, pull in those stats too. 

<code>AutoLagRollStatsScoring()</code> builds the above features for a partial set of records in a data set. The function is extremely useful as it can compute these feature vectors at a significantly faster rate than the non scoring version which comes in handy for scoring ML models. If you can find a way to make it faster, let me know.

</p>
</details>



#### **AutoLagRollMode()**

<details><summary>Code Example</summary>
<p>
 
```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoLagRollMode()</code> Generate lags and rolling modes for categorical variables
 
</p>
</details>



#### **AutoDiffLagN()**

<details><summary>Code Example</summary>
<p>
 
```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoDiffLagN()</code> Generate differences for numeric columns, date columns, and categorical columns, by groups. You can specify NLag1 and NLag2 to generate the diffs based on any two time periods.
 
</p>
</details>

#### **AutoInteraction()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>AutoInteraction()</code> will build out any number of interactions you want for numeric variables. You supply a character vector of numeric or integer column names, along with the names of any numeric columns you want to skip (including the interaction column names) and the interactions will be automatically created for you. For example, if you want a 4th degree interaction from 10 numeric columns, you will have 10 C 2, 10 C 3, and 10 C 4 columns created. Now, let's say you build all those features and decide you don't want all 10 features to be included. Remove the feature name from the NumericVars character vector. Now, let's say you modeled all of the interaction features and want to remove the ones will the lowest scores on the variable importance list. Grab the names and run the interaction function again except this time supply those poor performing interaction column names to the SkipCols argument and they will be ignored. Now, if you want to interact any categorical variable with a numeric variable, you'll have to dummify the categorical variable first and then include the level specific dummy variable column names to the NumericVars character vector argument. If you set Center and Scale to TRUE then the interaction multiplication won't create huge numbers.

</p>
</details>

#### **AutoWord2VecModeler()** and **AutoWord2VecScoring()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>AutoWord2VecModeler()</code> generates a specified number of vectors (word2vec) for each column of text data in your data set that you specify and it will save the models if you specify for re-creating them later in a model scoring process. You can choose to build individual models for each column or one model for all your columns. If you need to run several models for groups of text variables you can run the function several times. 

<code>AutoWord2VecScoring()</code> this is for generating word2vec vectors for model scoring situations. The function will load the model, create the transformations, and merge them onto the source data.table just like the training version does.

</p>
</details>

#### **CategoricalEncoding()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>CategoricalEncoding()</code> enables you to convert your categorical variables into numeric variables in seven different ways. You can choose from m_estimator, credibility (a.k.a. James Stein), weight of evidence, target encoding, poly encoding, backward difference encoding, and helmert encoding. You can run the function for training data and for scoring situations (on demand or batch). For scoring, you can choose to supply an imputation value for new levels that may show up or you can manage them somewhere else in the pipeline. For scoring, you have two options: during the training run you can save the metadata to file by supplying a path to SavePath or you can have the metadata returned by setting ReturnFactorLevelList to TRUE and in scoring your can either have the files pulled from file using the SavePath argument and the function will take care of the rest or you can supply the ReturnFactorLevelList to the SupplyFactorLevelList argument and the function will take care of the rest.

</p>
</details>


#### **H2OAutoencoder()** and **H2OAutoencoderScoring()**

<details><summary>Code Example</summary>
<p>


```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>H2OAutoencoder()</code> Use for dimension reduction and anomaly detection

<code>H2OAutoencoderScoring()</code> Use for dimension reduction and anomaly detection scoring

</p>
</details>

#### **H2OIsolationForest()** and **H2OIsolationForestScoring()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>H2OIsolationForecast()</code> Anomaly detection and feature engineering using H2O Isolation Forest. A model is built, your training data is scored, and the model is saved to file for later use in scoring environments with H2OIsolationForestScoring()

<code>H2OIsolationForecastScoring()</code> Scoring function

</p>
</details>

#### **AutoClustering() and AutoClusteringScoring()** 

<details><summary>Code Example</summary>
<p>

```

```

<details><summary>Function Description</summary>
<p>
 
<code>AutoClustering()</code> Generates a single column and merges it onto your data. You can have an autoencoder ran to reduce the dimension size before running the KMeans grid tuning operation. If you provide a directory path, the models will be saved and can be used later in scoring enviroments. I find that I utilize clustering more often for feature engineering that unsupervised learning which is why I put the code example and description here. The function utilizes H2O under the hood with their KMeans algo for the clustering and their deep learning algo for the dimensionality reduction. 

</p>
</details>

</p>
</details>

#### **CreateCalendarVariables()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>CreateCalendarVariables()</code> This functions creates numerical columns based on the date columns you supply such as second, minute, hour, week day, day of month, day of year, week, isoweek, wom, month, quarter, and year.

</p>
</details>

#### **CreateHolidayVariable()**

<details><summary>Code Example</summary>
<p>
 
```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>
 
<code>CreateHolidayVariable()</code> 
This function counts up the number of specified holidays between the current record time stamp and the previous record time stamp, by group as well if specified.

</p>
</details>

#### **DummifyDT()** 

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>DummifyDT()</code> This function is used in the AutoXGBoost__() suite of modeling functions to manage categorical variables in your training, validation, and test sets. This function rapidly dichotomizes categorical columns in a data.table (N+1 columns for N levels using one hot encoding or N columns for N levels otherwise). Several other arguments exist for outputting and saving factor levels. This is useful in model training, validating, and scoring processes.

</p>
</details>

#### **AutoDataPartition()**

<details><summary>Code Example</summary>
<p>

```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>AutoDataPartition()</code> is designed to achieve a few things that standard data partitioning processes or functions don't handle. First, you can choose to build any number of partitioned data sets beyond the standard train, validate, and test data sets. Second, you can choose between random sampling to split your data or you can choose a time-based partitioning. Third, for the random partitioning, you can specify a stratification columns in your data to stratify by in order to ensure a proper split amongst your categorical features (E.g. think MultiClass targets). Lastly, it's 100% data.table so it will run fast and with low memory overhead.

</p>
</details>

#### **ModelDataPrep()**

<details><summary>Code Example</summary>
<p>
 
```

```

</p>
</details>

<details><summary>Function Description</summary>
<p>

<code>ModelDataPrep()</code> This function will loop through every column in your data and apply a variety of functions based on argument settings. For all columns not ignored, these tasks include:
* Character type to Factor type converstion
* Factor type to Character type conversion
* Constant value imputation for numeric and categorical columns
* Integer type to Numeric type conversion
* Date type to Character type conversion
* Remove date columns
* Ignore specified columns

</p>
</details>

#### **AutoTransformationCreate()** and **AutoTransformationScore()**

<details><summary>Function Description</summary>
<p>
 
<code>AutoTransformationCreate()</code> is a function for automatically identifying the optimal transformations for numeric features and transforming them once identified. This function will loop through your selected transformation options (YeoJohnson, BoxCox, Asinh, Log, LogPlus1, Sqrt, along with Asin and Logit for proportion data) and find the one that produces the best fit to a normal distribution. It then generates the transformation and collects the metadata information for use in the AutoTransformationScore() function, either by returning the objects or saving them to file.

<code>AutoTransformationScore()</code> is a the compliment function to AutoTransformationCreate(). Automatically apply or inverse the transformations you identified in AutoTransformationCreate() to other data sets. This is useful for applying transformations to your validation and test data sets for modeling, which is done automatically for you if you specify.

</p>
</details>

#### **AutoHierarchicalFourier()**

<details><summary>Function Description</summary>
<p>
 
<code>AutoHierarchicalFourier()</code> turns time series data into fourier series. This function can generate any number of fourier pairs the user wants (if they can actually build) and you can run it with grouped time series data. In the grouping case, fourier pairs can be created for each categorical variable along with the full interactions between specified categoricals. The process is parallelized as well to run as fast as possible.

</p>
</details>

</p>
</details>
