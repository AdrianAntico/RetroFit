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
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
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
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
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
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
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
# Ftrl Example
####################################

# Setup Environment
import timeit
import datatable as dt
from datatable import sort, f, by
import retrofit
from retrofit import FeatureEngineering as fe
from retrofit import MachineLearning as ml

# Load some data
# BechmarkData.csv is located is the tests folder
Path = "C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv"
data = dt.fread(Path)

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
  TargetColumnName = 'Leads',
  NumericColumnNames = ['XREGS1', 'XREGS2', 'XREGS3'],
  CategoricalColumnNames = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'],
  TextColumnNames = None,
  WeightColumnName = None,
  Threads = -1,
  InputFrame = 'datatable')

# Get args list for algorithm and target type
ModelArgs = ml.ML0_Parameters(
  Algorithms = 'Ftrl', 
  TargetType = "Regression", 
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


