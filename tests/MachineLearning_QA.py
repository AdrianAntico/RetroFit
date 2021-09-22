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
  TargetType = "Regression", 
  TrainMethod = "Train")

# Initialize RetroFit
x = ml.RetroFit(
  ModelArgs,
  ModelData,
  DataFrames)

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

FitName = x.FitListNames[0]

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

x.ML1_Single_Evaluate(FitName=x.FitListNames[0], TargetType='classification', ScoredDataName=x.DataSetsNames[4], ByVariables=None)

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
  TargetType = "MultiClass",
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
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
  CategoricalColumnNames = ['Factor_1', 'Factor_2', 'Factor_3'],
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
  NumericColumnNames = [z for z in list(data.names) if z not in ['Factor_1','Factor_2','Factor_3','Adrian']],
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
x = ml.RetroFit(
  ModelArgs,
  ModelData,
  DataFrames)

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
x = ml.RetroFit(
  ModelArgs,
  ModelData,
  DataFrames)

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
  TargetType = "Classification", 
  TrainMethod = "Train")

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
  TargetType = "MultiClass",
  TrainMethod = "Train")

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

# Evaluate scored data
metrics = x.ML1_Single_Evaluate(
  FitName=x.FitListNames[0],
  TargetType=x.ModelArgs.get('LightGBM')['TargetType'],
  ScoredDataName=x.DataSetsNames[-1],
  ByVariables=None,
  CostDict=None)

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
  TargetType = "Classification", 
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
  TargetType = "MultiClass", 
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

