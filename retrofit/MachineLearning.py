# Module: MachineLearning
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.0.9
# Last modified : 2021-09-02

def ML0_GetModelData(TrainData=None, ValidationData=None, TestData=None, ArgsList=None, TargetColumnName=None, NumericColumnNames=None, CategoricalColumnNames=None, TextColumnNames=None, WeightColumnName=None, Threads=-1, Processing='catboost', InputFrame='datatable'):
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

    # ML0_GetModelData Example:
    import timeit
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
    t_start = timeit.default_timer()
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
      
    # timer
    t_end = timeit.default_timer()
    t_end - t_start
    
    # Collect catboost training data
    catboost_train = DataSets['train_data']
    catboost_validation = DataSets['validation_data']
    catboost_test = DataSets['test_data']
    ArgsList = DataSets['ArgsList']
    
    # QA: Group Case: Step through function
    TrainData=TrainData
    ValidationData=ValidationData
    TestData=TestData
    ArgsList=None
    TargetColumnName='Leads'
    NumericColumnNames=['XREGS1','XREGS2','XREGS3']
    CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    TextColumnNames=None
    WeightColumnName=None
    Threads=-1
    Processing='catboost'
    InputFrame='datatable'
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
    if InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import sort, f, by, ifelse

    # Import polars methods
    if InputFrame.lower() == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)
    
    # Convert to list if not already
    if not NumericColumnNames is None and not isinstance(NumericColumnNames, list):
      NumericColumnNames = [NumericColumnNames]
    if not CategoricalColumnNames is None and not isinstance(CategoricalColumnNames, list):
      CategoricalColumnNames = [CategoricalColumnNames]
    if not TextColumnNames is None and not isinstance(TextColumnNames, list):
      TextColumnNames = [TextColumnNames]
    
    # CatBoost
    if Processing.lower() == 'catboost':
      
      # Imports
      from catboost import Pool
      
      # label
      TrainLabel = TrainData[TargetColumnName].to_numpy()
      if not ValidationData is None:
        ValidationLabel = ValidationData[TargetColumnName].to_numpy()
      if not TestData is None:
        TestLabel = TestData[TargetColumnName].to_numpy()

      # data (numeric features)
      if not NumericColumnNames is None:
        SD = copy.copy(NumericColumnNames)
      else:
        SD = []
      if not CategoricalColumnNames is None:
        for nam in CategoricalColumnNames:
          SD.append(nam)
      if not TextColumnNames is None:
        for nam in TextColumnNames:
          SD.append(nam)
      if not WeightColumnName is None:
        SD.append(WeightColumnName)

      # data
      train = TrainData[:, SD].to_pandas()
      if not ValidationData is None:
        validation = ValidationData[:, SD].to_pandas()
      if not TestData is None:
        test = TestData[:, SD].to_pandas()

      # label
      trainlabel = TrainData[:, TargetColumnName].to_pandas()
      if not ValidationData is None:
        validationlabel = ValidationData[:, TargetColumnName].to_pandas()
      if not TestData is None:
        testlabel = TestData[:, TargetColumnName].to_pandas()

      # TrainData
      train_data = Pool(
        data =  train,
        label = trainlabel,
        cat_features = CategoricalColumnNames,
        text_features = TextColumnNames, 
        weight=WeightColumnName, 
        thread_count=Threads,
        pairs=None, has_header=False, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, feature_names=None)

      # ValidationData
      if not ValidationData is None:
        validation_data = Pool(
          data =  validation,
          label = validationlabel,
          cat_features = CategoricalColumnNames,
          text_features = TextColumnNames, 
          weight=WeightColumnName,
          thread_count=Threads,
          pairs=None, has_header=False, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, feature_names=None)
          
      # TestData
      if not TestData is None:
        test_data = Pool(
          data =  test,
          label = testlabel,
          cat_features = CategoricalColumnNames,
          text_features = TextColumnNames, 
          weight=WeightColumnName,
          thread_count=Threads,
          pairs=None, has_header=False, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, feature_names=None)
    
      # Return catboost
      return dict(train_data=train_data, validation_data=validation_data, test_data=test_data, ArgsList=ArgsList)
