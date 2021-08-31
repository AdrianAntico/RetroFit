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