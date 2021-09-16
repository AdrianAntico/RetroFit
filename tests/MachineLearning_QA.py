############################################################################################
# ML0_GetModelData Example
############################################################################################

import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# CatBoost

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

# QA: Group Case: Step through function
# TrainData=TrainData
# ValidationData=ValidationData
# TestData=TestData
# ArgsList=None
# TargetColumnName='Leads'
# NumericColumnNames=['XREGS1','XREGS2','XREGS3']
# CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
# TextColumnNames=None
# WeightColumnName=None
# Threads=-1
# Processing='catboost'
# InputFrame='datatable'

# XGBoost

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

# QA: Group Case: Step through function
# TrainData=TrainData
# ValidationData=ValidationData
# TestData=TestData
# ArgsList=None
# TargetColumnName='Leads'
# NumericColumnNames=['XREGS1','XREGS2','XREGS3']
# CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
# TextColumnNames=None
# WeightColumnName=None
# Threads=-1
# Processing='xgboost'
# InputFrame='datatable'


# LightGBM

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

# QA: Group Case: Step through function
# TrainData=TrainData
# ValidationData=ValidationData
# TestData=TestData
# ArgsList=None
# TargetColumnName='Leads'
# NumericColumnNames=['XREGS1','XREGS2','XREGS3']
# CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
# TextColumnNames=None
# WeightColumnName=None
# Threads=-1
# Processing='lightgbm'
# InputFrame='datatable'


############################################################################################
# ML0_Parameters
############################################################################################

# Ftrl
Params = ML0_Parameters(Algorithm='Ftrl', TargetType='regression', TrainMethod='gridtune', Model=None)

print(Params)
print_list(Params)

############################################################################################
# RetroFit Class
############################################################################################

####################################
# Ftrl Regression
####################################

# Setup Environment
import pkg_resources
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering_old as fe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/ClassificationData.csv')
data = dt.fread(FilePath)

# Create partitioned data sets
DataFrames = fe.FE2_AutoDataParition(
  data = data, 
  ArgsList = None, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False, 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'Ftrl',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Adrian',
  NumericColumnNames = list(data.names[1:11]),
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'Ftrl', 
  TargetType = "Classification", 
  TrainMethod = "Train")

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

####################################
# CatBoost Example Usage
####################################

# Setup Environment
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
DataFrames = fe.FE2_AutoDataParition(
  data = data, 
  ArgsList = None, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False, 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'catboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Leads',
  NumericColumnNames = ['XREGS1', 'XREGS2', 'XREGS3'],
  CategoricalColumnNames = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'CatBoost', 
  TargetType = "Regression", 
  TrainMethod = "Train")

# Update iterations to run quickly
ModelArgs['CatBoost']['AlgoArgs']['iterations'] = 50

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


####################################
# XGBoost Example Usage
####################################

# Setup Environment
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Dummify
Output = fe.FE1_DummyVariables(
  data = data, 
  ArgsList = None, 
  CategoricalColumnNames = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3'], 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')
data = Output['data']
data = data[:, [name not in ['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'] for name in data.names]]

# Create partitioned data sets
DataFrames = fe.FE2_AutoDataParition(
  data = data, 
  ArgsList = None, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False, 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')

# Features
Features = ['XREGS1', 'XREGS2', 'XREGS3', 'MarketingSegments_B', 'MarketingSegments_A', 'MarketingSegments_C', 'MarketingSegments2_a', 'MarketingSegments2_b', 'MarketingSegments2_c', 'MarketingSegments3_x', 'MarketingSegments3_z', 'MarketingSegments3_y']

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'xgboost',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Leads',
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

####################################
# LightGBM Example Usage
####################################

# Setup Environment
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# Load some data
FilePath = pkg_resources.resource_filename('retrofit', 'datasets/BenchmarkData.csv') 
data = dt.fread(FilePath)

# Dummify
Output = fe.FE1_DummyVariables(
  data = data, 
  ArgsList = None, 
  CategoricalColumnNames = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3'], 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')
data = Output['data']
data = data[:, [name not in ['MarketingSegments','MarketingSegments2','MarketingSegments3','Label'] for name in data.names]]

# Create partitioned data sets
DataFrames = fe.FE2_AutoDataParition(
  data = data, 
  ArgsList = None, 
  DateColumnName = None, 
  PartitionType = 'random', 
  Ratios = [0.7,0.2,0.1], 
  ByVariables = None, 
  Sort = False, 
  Processing = 'datatable', 
  InputFrame = 'datatable', 
  OutputFrame = 'datatable')

# Features
Features = ['XREGS1', 'XREGS2', 'XREGS3', 'MarketingSegments_B', 'MarketingSegments_A', 'MarketingSegments_C', 'MarketingSegments2_a', 'MarketingSegments2_b', 'MarketingSegments2_c', 'MarketingSegments3_x', 'MarketingSegments3_z', 'MarketingSegments3_y']

# Prepare modeling data sets
ModelData = ml.ML0_GetModelData(
  Processing = 'lightgbm',
  TrainData = DataFrames['TrainData'],
  ValidationData = DataFrames['ValidationData'],
  TestData = DataFrames['TestData'],
  ArgsList = None,
  TargetColumnName = 'Leads',
  NumericColumnNames = Features,
  CategoricalColumnNames = None,
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'LightGBM', 
  TargetType = "Regression", 
  TrainMethod = "Train")

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
