# Module: DataPrep
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.0.4
# Last modified : 2021-08-29

def AutoDataParition(data=None, ArgsList=None, DateColumnName=None, PartitionType='random', Ratios=None, Dates=None, ByVariables=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    
    """
    # Goal:
    Automatically generate train, validation, and test data sets for modeling purposes
      
    # Output
    Return a datatable, polars frame, or pandas frame with new lag columns
    
    # Parameters
    data:           Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:       None or Dict. If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    DateColumnName: Scalar. Primary date column used for sorting
    PartitionType:  Scalar. Columns to partition over
    Ratios:         List. Use ths for PartitionType 'random'. List of decimal values for determining how many data goes into each data frame.
    Dates:          List. Use ths for PartitionType 'time'. List of dates for splitting. 1 date for two frames, 2 dates for 3 frames.
    ByVariables:    None or List. Stratify the data paritioning using ByVariables
    Processing:     'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:     'datatable', 'polars', or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    'datatable', 'polars', or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # Example
    import datatable as dt
    import retrofit
    from retrofit import DataPrep as dp
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    Output = AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='random', Ratios=[0.70,0.20,0.10], Dates=None, ByVariables=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    Output = dp.AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='random', Ratios=[0.70,0.20,0.10], Dates=None, ByVariables=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    TrainData = Output['TrainData']
    ValidationData = Output['ValidationData']
    TestData = Output['TestData']
    
    # QA
    ArgsList=None
    DateColumnName='CalendarDateColumn'
    PartitionType='random'
    Ratios=[0.70,0.20,0.10]
    Dates=None
    ByVariables=None
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    
    """
  
    # ArgsList Collection
    if not ArgsList is None:
      DateColumnName = ArgsList['DateColumnName']
      PartitionType = ArgsList['PartitionType']
      Ratios = ArgsList['Ratios']
      Dates = ArgsList['Dates']
      ByVariables = ArgsList['ByVariables']
    else :
      ArgsList = dict(
        DateColumnName=DateColumnName,
        PartitionType=PartitionType,
        Ratios=Ratios,
        Dates=Dates,
        ByVariables=ByVariables)

    # For making copies of lists so originals aren't modified
    import numpy as np

    # Import datatable methods
    if Processing == 'datatable' or OutputFrame == 'datatable' or InputFrame == 'datatable':
      import datatable as dt
      from datatable import f, by

    # Import polars methods
    if Processing == 'polars' or OutputFrame == 'polars' or InputFrame == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Convert to datatable
    if InputFrame == 'pandas' and Processing == 'datatable':
      data = dt.Frame(data)
    elif InputFrame == 'pandas' and Processing == 'polars':
      data = pl.from_pandas(data)

    # Accumulate Ratios
    Ratios = cumsum(Ratios)

    # Random partitioning
    if PartitionType == 'random':

      # Add random number column
      data = data[:, f[:].extend({"ID": np.random.uniform(0,1, size = data.shape[0])})]

      # TrainData
      TrainData = data[f.ID <= Ratios[0], ...]
      del TrainData['ID']

      # ValidationData
      ValidationData = data[(f.ID <= Ratios[1]) & (f.ID > Ratios[0]), ...]
      del ValidationData['ID']

      # TestData
      if len(Ratios) >= 3:
        TestData = data[f.ID > Ratios[1], ...]
        del TestData['ID']
      else:
        TestData = None

    # Time base partitioning
    if PartitionType == 'time':

      # TrainData
      TrainData = data[f[DateColumnName] <= Dates[0], ...]
      del TrainData[f[DateColumnName]]

      # ValidationData
      ValidationData = data[(f[DateColumnName] <= Dates[1]) & (f[DateColumnName] > Dates[0]), ...]
      del ValidationData[f[DateColumnName]]

      # TestData
      if len(Dates) == 2:
        TestData = data[f[DateColumnName] > Dates[1], ...]
        del TestData[f[DateColumnName]]
      else:
        TestData = None

    return dict(TrainData = TrainData, ValidationData = ValidationData, TestData = TestData)



def ModelDataPrepare(TrainData=None, ValidationData=None, TestData=None, ArgsList=None, TargetColumnName=None, NumericColumnNames=None, CategoricalColumnNames=None, TextColumnNames=None, WeightColumnName=None, Threads=-1, Processing='catboost', InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Create modeling objects for specific algorithms. E.g. create train, valid, and test objects for catboost
    
    # Output
    Return frames for catboost, xgboost, lightgbm, etc.
    
    # Parameters
    TrainData:              Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ValidationData:         Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    TestData:               Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:               If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    TargetColumnName:       A list of columns that will be lagged
    NumericColumnNames:     Primary date column used for sorting
    CategoricalColumnNames: Columns to partition over
    TextColumnNames:        List of integers for the lookback lengths
    WeightColumnName:       Value to fill the NA's for beginning of series
    Threads:                Number of threads to utilize if available for the algorithm
    Processing:             'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:             'datatable', 'polars', or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:            'datatable', 'polars', or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test ModelDataPrepare
    import datatable as dt
    from datatable import sort, f, by
    import retrofit
    from retrofit import TimeSeriesFeatures as ts

    # Example:
    
    
    # QA: Group Case: Step through function
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    LagPeriods = 1
    LagColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    lp = 1
    ImputeValue = -1
    Sort = True
    """
    
    # ArgsList Collection
    if not ArgsList is None:
      TargetColumnName = ArgsList['TargetColumnName']
      NumericColumnNames = ArgsList['NumericColumnNames']
      CategoricalColumnNames = ArgsList['CategoricalColumnNames']
      TextColumnNames = ArgsList['TextColumnNames']
      WeightColumnName = ArgsList['WeightColumnName']
      Threads = ArgsList['Threads'],
      Processing = ArgsList['Processing']
    else:
      ArgsList = dict(
        TargetColumnName=TargetColumnName,
        NumericColumnNames=NumericColumnNames,
        CategoricalColumnNames=CategoricalColumnNames,
        TextColumnNames=TextColumnNames,
        WeightColumnName=WeightColumnName,
        Threads=Threads,
        Processing=Processing)

    # For making copies of lists so originals aren't modified
    import copy
    
    # Import datatable methods
    if Processing == 'datatable' or OutputFrame == 'datatable' or InputFrame == 'datatable':
      import datatable as dt
      from datatable import sort, f, by, ifelse

    # Import polars methods
    if Processing == 'polars' or OutputFrame == 'polars' or InputFrame == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Convert to datatable
    if InputFrame == 'pandas' and Processing == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame == 'pandas' and Processing == 'polars':
      data = pl.from_pandas(data)
    
    # CatBoost
    if Processing.to_lower() == 'catboost':
      
      # TrainData
      train_data = Pool(
        data = TrainData,
        label=train_target, 
        cat_features=train_categorical, 
        text_features=train_text, 
        pairs=None,
        delimiter='\t',
        has_header=False,
        weight=WeightColumnName, 
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None
        baseline=None,
        feature_names=None,
        thread_count=-1)
      
      # ValidationData
      if not ValidationData is None:
        validation_data = Pool(
          data = TrainData,
          label=Train_Target, 
          cat_features=Train_Categorical, 
          text_features=Train_Text, 
          pairs=None,
          delimiter='\t',
          has_header=False,
          weight=None, 
          group_id=None,
          group_weight=None,
          subgroup_id=None,
          pairs_weight=None
          baseline=None,
          feature_names=None,
          thread_count=-1)
          
      # TestData
      if not TestData is None:
        test_data = Pool(
          data = TestData,
          label=Train_Target, 
          cat_features=Train_Categorical, 
          text_features=Train_Text, 
          pairs=None,
          delimiter='\t',
          has_header=False,
          weight=None, 
          group_id=None,
          group_weight=None,
          subgroup_id=None,
          pairs_weight=None
          baseline=None,
          feature_names=None,
          thread_count=-1)
    
    # Return
    return dict(train_data=train_data, validation_data=validation_data, test_data=test_data)
