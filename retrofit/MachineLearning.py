# Module: MachineLearning
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.3
# Last modified : 2021-09-14

def ML0_GetModelData(TrainData=None, ValidationData=None, TestData=None, ArgsList=None, TargetColumnName=None, NumericColumnNames=None, CategoricalColumnNames=None, TextColumnNames=None, WeightColumnName=None, Threads=-1, Processing='catboost', InputFrame='datatable'):
    """
    # Goal:
    Create modeling objects for specific algorithms. E.g. create train, valid, and test objects for catboost
    
    # Output
    Return frames for catboost, xgboost, and lightgbm, currently.
    
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
    Processing:             'catboost', 'xgboost', 'lightgbm', or 'ftrl'
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

    # Ftrl
    if Processing.lower() == 'ftrl':
      
      # data (numeric features)
      if not NumericColumnNames is None:
        SD = copy.copy(NumericColumnNames)
      else:
        SD = []
      if not CategoricalColumnNames is None:
        SD.extend(CategoricalColumnNames)
      if not TextColumnNames is None:
        SD.extend(TextColumnNames)

      # TrainData
      train_data = TrainData[:, SD]
      validation_data = ValidationData[:, SD]
      test_data = TestData[:, SD]

      # Return catboost
      return dict(train_data=TrainData, validation_data=ValidationData, test_data=TestData, ArgsList=ArgsList)
    
    # CatBoost
    if Processing.lower() == 'catboost':
      
      # Imports
      from catboost import Pool

      # data (numeric features)
      if not NumericColumnNames is None:
        SD = copy.copy(NumericColumnNames)
      else:
        SD = []
      if not CategoricalColumnNames is None:
        SD.extend(CategoricalColumnNames)
      if not TextColumnNames is None:
        SD.extend(TextColumnNames)
      if not WeightColumnName is None:
        SD.extend(WeightColumnName)

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

    # XGBoost
    if Processing.lower() == 'xgboost':
      
      # Imports
      import xgboost as xgb

      # data (numeric features)
      if not NumericColumnNames is None:
        SD = copy.copy(NumericColumnNames)
      else:
        SD = []
      if not WeightColumnName is None:
        trainweightdata = TrainData['WeightColumnName'].to_pandas()
        if not ValidationData is None:
          validationweightdata = ValidationData['WeightColumnName'].to_pandas()
        if not TestData is None:
          testweightdata = TestData['WeightColumnName'].to_pandas()
      else:
        trainweightdata = None
        validationweightdata = None
        testweightdata = None
        
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
      if trainweightdata is None:
        train_data = xgb.DMatrix(data = train, label = trainlabel)
      else:
        train_data = xgb.DMatrix(data = train, label = trainlabel, weight = trainweightdata)
      
      # ValidationData
      if not ValidationData is None:
        if validationweightdata is None:
          validation_data = xgb.DMatrix(data = validation, label = validationlabel)
        else:
          validation_data = xgb.DMatrix(data = validation, label = validationlabel, weight = validationweightdata)
        
      # TestData
      if not TestData is None:
        if testweightdata is None:
          test_data = xgb.DMatrix(data = test, label = testlabel)
        else:
          test_data = xgb.DMatrix(data = test, label = testlabel, weights = testweightdata)
    
      # Return catboost
      return dict(train_data=train_data, validation_data=validation_data, test_data=test_data, ArgsList=ArgsList)
    
    # LightGBM
    if Processing.lower() == 'lightgbm':
      
      # Imports
      import lightgbm as lgbm

      # data (numeric features)
      if not NumericColumnNames is None:
        SD = copy.copy(NumericColumnNames)
      else:
        SD = []
      if not WeightColumnName is None:
        trainweightdata = TrainData['WeightColumnName'].to_pandas()
        if not ValidationData is None:
          validationweightdata = ValidationData['WeightColumnName'].to_pandas()
        if not TestData is None:
          testweightdata = TestData['WeightColumnName'].to_pandas()
      else:
        trainweightdata = None
        validationweightdata = None
        testweightdata = None
        
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
      if trainweightdata is None:
        train_data = lgbm.Dataset(data = train, label = trainlabel)
      else:
        train_data = lgbm.Dataset(data = train, label = trainlabel, weight = trainweightdata)
      
      # ValidationData
      if not ValidationData is None:
        if validationweightdata is None:
          validation_data = lgbm.Dataset(data = validation, label = validationlabel)
        else:
          validation_data = lgbm.Dataset(data = validation, label = validationlabel, weight = validationweightdata)
        
      # TestData
      if not TestData is None:
        if testweightdata is None:
          test_data = lgbm.Dataset(data = test, label = testlabel)
        else:
          test_data = lgbm.Dataset(data = test, label = testlabel, weights = testweightdata)
    
      # Return catboost
      return dict(train_data=train_data, validation_data=validation_data, test_data=test_data, ArgsList=ArgsList)


def ML0_Parameters(Algorithms=None, TargetType=None, TrainMethod=None, Model=None):
    """
    # Goal
    Return an ArgsList appropriate for the algorithm selection, target type, and training method
    
    # Parameters
    Algorithms:       Choose from CatBoost, XGBoost, LightGBM, Ftrl
    TargetType:       Choose from 'regression', 'classification', 'multiclass'
    TrainMethod:      Choose from 'train', 'gridtune'
    GetModelDataArgs: Args passed in from ML0_GetModelData()
    
    # ML0_Parameters Example
    import timeit
    import datatable as dt
    from datatable import sort, f, by
    import retrofit
    from retrofit import FeatureEngineering as fe
    from retrofit import MachineLearning as ml

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

    # Collect Args
    Args = DataSets.get('ArgsList')

    # Create Parameters for Modeling
    ModelArgs = ml.ML0_Parameters(
      Algorithms='catboost', 
      TargetType='regression', 
      TrainMethod='Train', 
      Model=None)

    # QA
    Algorithms='catboost'
    TargetType='regression'
    TrainMethod='Train'
    Model=None
    Algo = 'catboost'
    """
    
    # Ensure Algorithms is a list
    if not isinstance(Algorithms, list):
      Algorithms = [Algorithms]

    # Loop through algorithms
    MasterArgs = dict()
    for Algo in Algorithms:
      
      # Initialize ArgsList
      ArgsList = {}
      ArgsList['Algorithms'] = Algo
      ArgsList['TargetType'] = TargetType
      ArgsList['TrainMethod'] = TrainMethod
    
      #############################################
      # Algorithm Selection CatBoost
      #############################################
      if Algo.lower() == 'catboost':

        # Setup Environment
        import catboost as cb
        import os

        # Initialize AlgoArgs
        AlgoArgs = dict()

        ###############################
        # TargetType Parameters
        ###############################
        if ArgsList.get('TargetType').lower() == 'classification':
          AlgoArgs['auto_class_weights'] = 'Balanced'
          AlgoArgs['loss_function'] = 'CrossEntropy'
          AlgoArgs['eval_metric'] = 'CrossEntropy'
        elif ArgsList.get('TargetType').lower() == 'multiclass':
          AlgoArgs['classes_count'] = 3
          AlgoArgs['loss_function'] = 'MCC'
          AlgoArgs['eval_metric'] = 'MCC'
        elif ArgsList.get('TargetType').lower() == 'regression':
          AlgoArgs['loss_function'] = 'RMSE'
          AlgoArgs['eval_metric'] = 'RMSE'

        ###############################
        # Parameters
        ###############################
        AlgoArgs['train_dir'] = os.getcwd()
        AlgoArgs['task_type'] = 'GPU'
        AlgoArgs['learning_rate'] = None
        AlgoArgs['l2_leaf_reg'] = None
        AlgoArgs['has_time'] = False
        AlgoArgs['best_model_min_trees'] = 10
        AlgoArgs['nan_mode'] = 'Min'
        AlgoArgs['fold_permutation_block'] = 1
        AlgoArgs['boosting_type'] = 'Plain'
        AlgoArgs['random_seed'] = None
        AlgoArgs['thread_count'] = -1
        AlgoArgs['metric_period'] = 10

        ###############################
        # Gridable Parameters
        ###############################
        if TrainMethod.lower() == 'train':
          AlgoArgs['iterations'] = 1000
          AlgoArgs['depth'] = 6
          AlgoArgs['langevin'] = True
          AlgoArgs['diffusion_temperature'] = 10000
          AlgoArgs['grow_policy'] = 'SymmetricTree'
          AlgoArgs['model_size_reg'] = 0.5
        else:
          AlgoArgs['iterations'] = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
          AlgoArgs['depth'] = [4, 5, 6, 7, 8, 9, 10]
          AlgoArgs['langevin'] = [True, False]
          AlgoArgs['diffusion_temperature'] = [7500, 10000, 12500]
          AlgoArgs['grow_policy'] = ['SymmetricTree', 'Lossguide', 'Depthwise']
          AlgoArgs['model_size_reg'] = [0.0, 0.25, 0.5, 0.75, 1.0]

        ###############################
        # Dependent Model Parameters
        ###############################

        # task_type dependent
        if AlgoArgs['task_type'] == 'GPU':
          AlgoArgs['bootstrap_type'] = 'Bayesian'
          AlgoArgs['score_function'] = 'L2'
          AlgoArgs['border_count'] = 128
        else:
          AlgoArgs['bootstrap_type'] = 'MVS'
          AlgoArgs['sampling_frequency'] = 'PerTreeLevel'
          AlgoArgs['random_strength'] = 1
          AlgoArgs['rsm'] = 0.80
          AlgoArgs['posterior_sampling'] = False
          AlgoArgs['score_function'] = 'L2'
          AlgoArgs['border_count'] = 254

        # Bootstrap dependent
        if AlgoArgs['bootstrap_type'] in ['Poisson', 'Bernoulli', 'MVS']:
          AlgoArgs['subsample'] = 1
        elif AlgoArgs['bootstrap_type'] in ['Bayesian']:
          AlgoArgs['bagging_temperature'] = 1

        # grow_policy
        if AlgoArgs['grow_policy'] in ['Lossguide', 'Depthwise']:
          AlgoArgs['min_data_in_leaf'] = 1
          if AlgoArgs['grow_policy'] == 'Lossguide':
            AlgoArgs['max_leaves'] = 31

        # boost_from_average
        if AlgoArgs['loss_function'] in ['RMSE', 'Logloss', 'CrossEntropy', 'Quantile', 'MAE', 'MAPE']:
          AlgoArgs['boost_from_average'] = True
        else:
          AlgoArgs['boost_from_average'] = False

        # Return
        ArgsList['AlgoArgs'] = AlgoArgs
        MasterArgs[Algo] = ArgsList

      #############################################
      # Algorithm Selection XGBoost
      #############################################
      if Algo.lower() == 'xgboost':
    
        # Setup Environment
        import xgboost as xgb
        import os
        AlgoArgs = dict()
        
        # Performance Params
        AlgoArgs['nthread'] = os.cpu_count()
        AlgoArgs['predictor'] = 'auto'
        AlgoArgs['single_precision_histogram'] = False
        AlgoArgs['early_stopping_rounds'] = 50
        
        # Training Params
        AlgoArgs['tree_method'] = 'gpu_hist'
        AlgoArgs['max_bin'] = 256
        
        ###############################
        # Gridable Parameters
        ###############################
        if TrainMethod.lower() == 'train':
          AlgoArgs['num_parallel_tree'] = 1
          AlgoArgs['num_boost_round'] = 1000 
          AlgoArgs['grow_policy'] = 'depthwise'
          AlgoArgs['eta'] = 0.30
          AlgoArgs['max_depth'] = 6
          AlgoArgs['min_child_weight'] = 1
          AlgoArgs['max_delta_step'] = 0
          AlgoArgs['subsample'] = 1.0
          AlgoArgs['colsample_bytree'] = 1.0
          AlgoArgs['colsample_bylevel'] = 1.0
          AlgoArgs['colsample_bynode'] = 1.0
          AlgoArgs['alpha'] = 0
          AlgoArgs['lambda'] = 1
          AlgoArgs['gamma'] = 0
        else:
          AlgoArgs['num_parallel_tree'] = [1, 5, 10]
          AlgoArgs['num_boost_round'] = [500, 1000, 1500, 2000, 2500]
          AlgoArgs['grow_policy'] = ['depthwise', 'lossguide']
          AlgoArgs['eta'] = [0.10, 0.20, 0.30]
          AlgoArgs['max_depth'] = [4, 5, 6, 7, 8]
          AlgoArgs['min_child_weight'] = [1, 5, 10]
          AlgoArgs['max_delta_step'] = [0, 1, 5, 10]
          AlgoArgs['subsample'] = [0.615, 0.8, 1]
          AlgoArgs['colsample_bytree'] = [0.615, 0.8, 1]
          AlgoArgs['colsample_bylevel'] = [0.615, 0.8, 1]
          AlgoArgs['colsample_bynode'] = [0.615, 0.8, 1]
          AlgoArgs['alpha'] = [0, 0.1, 0.2]
          AlgoArgs['lambda'] = [0.80, 0.90, 1.0]
          AlgoArgs['gamma'] = [0, 0.1, 0.5]

        # GPU Dependent
        if AlgoArgs['tree_method'] == 'gpu_hist':
          AlgoArgs['sampling_method'] = 'uniform'

        # Classification
        if ArgsList.get('TargetType').lower() == 'classification':
          AlgoArgs['objective'] = 'binary:logistic'
          AlgoArgs['eval_metric'] = 'auc'
        elif ArgsList.get('TargetType').lower() == 'regression':
          AlgoArgs['objective'] = 'reg:squarederror'
          AlgoArgs['eval_metric'] = 'rmse'
        elif ArgsList.get('TargetType').lower() == 'multiclass':
          AlgoArgs['objective'] = 'multi:softprob'
          AlgoArgs['eval_metric'] = 'mlogloss'

        # Return
        ArgsList['AlgoArgs'] = AlgoArgs
        MasterArgs[Algo] = ArgsList

      #############################################
      # Algorithm Selection LightGBM
      #############################################
      if Algo.lower() == 'lightgbm':
    
        # Setup Environment
        import os
        import lightgbm as lgbm
        AlgoArgs = dict()
        
        # Tuning Args
        if TrainMethod.lower() == 'train':
          AlgoArgs['num_iterations'] = 1000
          AlgoArgs['learning_rate'] = None
          AlgoArgs['num_leaves'] = 31
          AlgoArgs['bagging_freq'] = 0
          AlgoArgs['bagging_fraction'] = 1.0
          AlgoArgs['feature_fraction'] = 1.0
          AlgoArgs['feature_fraction_bynode'] = 1.0
          AlgoArgs['max_delta_step'] = 0.0
        else :
          AlgoArgs['num_iterations'] = [500, 1000, 1500, 2000, 2500]
          AlgoArgs['learning_rate'] = [0.05, 0.10, 0.15, 0.20, 0.25]
          AlgoArgs['num_leaves'] = [20, 25, 31, 36, 40]
          AlgoArgs['bagging_freq'] = [0.615, 0.80, 1.0]
          AlgoArgs['bagging_fraction'] = [0.615, 0.80, 1.0]
          AlgoArgs['feature_fraction'] = [0.615, 0.80, 1.0]
          AlgoArgs['feature_fraction_bynode'] = [0.615, 0.80, 1.0]
          AlgoArgs['max_delta_step'] = [0.0, 0.10 , 0.20]
        
        # Args
        AlgoArgs['task'] = 'train'
        AlgoArgs['device_type'] = 'CPU'
        AlgoArgs['objective'] = 'regression'
        AlgoArgs['metric'] = 'rmse'
        AlgoArgs['boosting'] = 'gbdt'
        AlgoArgs['lambda_l1'] = 0.0
        AlgoArgs['lambda_l2'] = 0.0
        AlgoArgs['deterministic'] = True
        AlgoArgs['force_col_wise'] = False
        AlgoArgs['force_row_wise'] = False
        AlgoArgs['max_depth'] = None
        AlgoArgs['min_data_in_leaf'] = 20
        AlgoArgs['min_sum_hessian_in_leaf'] = 0.001
        AlgoArgs['extra_trees'] = False
        AlgoArgs['early_stopping_round'] = 10
        AlgoArgs['first_metric_only'] = True
        AlgoArgs['linear_lambda'] = 0.0
        AlgoArgs['min_gain_to_split'] = 0
        AlgoArgs['monotone_constraints'] = None
        AlgoArgs['monotone_constraints_method'] = 'advanced'
        AlgoArgs['monotone_penalty'] = 0.0
        AlgoArgs['forcedsplits_filename'] = None
        AlgoArgs['refit_decay_rate'] = 0.90
        AlgoArgs['path_smooth'] = 0.0

        # IO Dataset Parameters
        AlgoArgs['max_bin'] = 255
        AlgoArgs['min_data_in_bin'] = 3
        AlgoArgs['data_random_seed'] = 1
        AlgoArgs['is_enable_sparse'] = True
        AlgoArgs['enable_bundle'] = True
        AlgoArgs['use_missing'] = True
        AlgoArgs['zero_as_missing'] = False
        AlgoArgs['two_round'] = False

        # Convert Parameters
        AlgoArgs['convert_model'] = None
        AlgoArgs['convert_model_language'] = 'cpp'

        # Objective Parameters
        AlgoArgs['boost_from_average'] = True
        AlgoArgs['alpha'] = 0.90
        AlgoArgs['fair_c'] = 1.0
        AlgoArgs['poisson_max_delta_step'] = 0.70
        AlgoArgs['tweedie_variance_power'] = 1.5
        AlgoArgs['lambdarank_truncation_level'] = 30

        # Metric Parameters (metric is in Core)
        AlgoArgs['is_provide_training_metric'] = True
        AlgoArgs['eval_at'] = [1,2,3,4,5]

        # Network Parameters
        AlgoArgs['num_machines'] = 1

        # GPU Parameters
        AlgoArgs['gpu_platform_id'] = -1
        AlgoArgs['gpu_device_id'] = -1
        AlgoArgs['gpu_use_dp'] = True
        AlgoArgs['num_gpu'] = 1

        # Return
        ArgsList['AlgoArgs'] = AlgoArgs
        MasterArgs[Algo] = ArgsList

      #############################################
      # Algorithm Selection Ftrl
      #############################################
      if Algo.lower() == 'ftrl':
    
        # Setup Environment
        import datatable
        from datatable.models import Ftrl
        AlgoArgs = dict()
    
        # TrainMethod Train
        model = Ftrl()
        AlgoArgs['interactions'] = model.interactions
        if TrainMethod.lower() == 'train':
          AlgoArgs['alpha'] = model.alpha
          AlgoArgs['beta'] = model.beta
          AlgoArgs['lambda1'] = model.lambda1
          AlgoArgs['lambda2'] = model.lambda2
          AlgoArgs['nbins'] = model.nbins
          AlgoArgs['mantissa_nbits'] = model.mantissa_nbits
          AlgoArgs['nepochs'] = model.nepochs
        else:
          AlgoArgs['alpha'] = [model.alpha, model.alpha * 2, model.alpha * 3]
          AlgoArgs['beta'] = [model.beta * 0.50, model.beta, model.beta * 1.5]
          AlgoArgs['lambda1'] = [model.lambda1, model.lambda1+0.05, model.lambda1+0.10]
          AlgoArgs['lambda2'] = [model.lambda2, model.lambda2+0.05, model.lambda2+0.10]
          AlgoArgs['nbins'] = [int(model.nbins*0.5), model.nbins, int(model.nbins*1.5)]
          AlgoArgs['mantissa_nbits'] = [int(model.mantissa_nbits / 2), model.mantissa_nbits, int(model.mantissa_nbits*1.5)]
          AlgoArgs['nepochs'] = [model.nepochs, model.nepochs*2, model.nepochs*3]
    
        # Target Type Specific Args
        if TargetType.lower() == 'regression':
          AlgoArgs['model_type'] = 'regression'
        elif TargetType.lower() == 'classification':
          AlgoArgs['model_type'] = 'binomial'
        elif TargetType.lower() == 'multiclass':
          AlgoArgs['negative_class'] = model.negative_class
          AlgoArgs['model_type'] = 'multinomial'

        # Return
        ArgsList['AlgoArgs'] = AlgoArgs
        MasterArgs[Algo] = ArgsList

    # Return
    return MasterArgs


# RetroFit Class 
class RetroFit:
    """
    ####################################
    # Goals
    ####################################
    
    Training
    Feature Tuning
    Grid Tuning
    Continued Training
    Scoring
    Model Evaluation
    Model Interpretation
    
    ####################################
    # Functions
    ####################################
    
    ML1_Single_Train()
    ML1_Single_Score()
    PrintAlgoArgs()
    
    ####################################
    # Attributes
    ####################################
    
    self.ModelArgs = ModelArgs
    self.ModelArgsNames = [*self.ModelArgs]
    self.Runs = len(self.ModelArgs)
    self.DataSets = DataSets
    self.DataSetsNames = [*self.DataSets]
    self.ModelList = dict()
    self.ModelListNames = []
    self.FitList = dict()
    self.FitListNames = []
    self.EvaluationList = dict()
    self.EvaluationListNames = []
    self.InterpretationList = dict()
    self.InterpretationListNames = []
    self.CompareModelsList = dict()
    self.CompareModelsListNames = []
    
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
    Path = "./BenchmarkData.csv"
    data = dt.fread(Path)
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")

    # Create partitioned data sets
    Data = fe.FE2_AutoDataParition(
      data=data, 
      ArgsList=None, 
      DateColumnName=None, 
      PartitionType='random', 
      Ratios=[0.7,0.2,0.1], 
      ByVariables=None, 
      Sort=False, 
      Processing='datatable', 
      InputFrame='datatable', 
      OutputFrame='datatable')

    # Prepare modeling data sets
    ModelData = ml.ML0_GetModelData(
      Processing='CatBoost',
      TrainData=Data['TrainData'],
      ValidationData=Data['ValidationData'],
      TestData=Data['TestData'],
      ArgsList=None,
      TargetColumnName='Leads',
      NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'],
      CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'],
      TextColumnNames=None,
      WeightColumnName=None,
      Threads=-1,
      InputFrame='datatable')

    # Get args list for algorithm and target type
    ModelArgs = ml.ML0_Parameters(
      Algorithms='CatBoost',
      TargetType='Regression',
      TrainMethod='Train')

    # Initialize RetroFit
    x = ml.RetroFit(ModelArgs, ModelData, DataFrames)

    # Train Model
    x.ML1_Single_Train(Algorithm='Ftrl')
    x.ML1_Single_Train(Algorithm='catboost')

    # Score data
    x.ML1_Single_Score(DataName=x.DataSetsNames[2], ModelName=x.ModelListNames[0])

    # Scoring data colnames
    x.DataSets['Scored_test_data'].names
    
    # Scoring data
    x.DataSets.get('Scored_test_data_Ftrl_1')

    # Check ModelArgs Dict
    x.ModelArgs

    # Check the names of data sets collected
    x.DataSetsNames

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
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
    # Create partitioned data sets
    DataFrames = fe.FE2_AutoDataParition(
      data=data, 
      ArgsList=None, 
      DateColumnName=None, 
      PartitionType='random', 
      Ratios=[0.7,0.2,0.1], 
      ByVariables=None, 
      Sort=False, 
      Processing='datatable', 
      InputFrame='datatable', 
      OutputFrame='datatable')
    
    # Prepare modeling data sets
    ModelData = ml.ML0_GetModelData(
      Processing='catboost',
      TrainData=DataFrames['TrainData'],
      ValidationData=DataFrames['ValidationData'],
      TestData=DataFrames['TestData'],
      ArgsList=None,
      TargetColumnName='Leads',
      NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'],
      CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'],
      TextColumnNames=None,
      WeightColumnName=None,
      Threads=-1,
      InputFrame='datatable')
    
    # Get args list for algorithm and target type
    ModelArgs = ml.ML0_Parameters(
      Algorithms='CatBoost', 
      TargetType="Regression", 
      TrainMethod="Train")
    
    # Initialize RetroFit
    x = ml.RetroFit(ModelArgs, ModelData, DataFrames)
    
    # Train Model
    x.ML1_Single_Train(Algorithm='CatBoost')
    
    # Score data
    x.ML1_Single_Score(DataName=x.DataSetsNames[2], ModelName=x.ModelListNames[0], Algorithm='CatBoost')
    
    # Scoring data colnames
    x.DataSets['Scored_test_data'].names
    
    # Scoring data
    x.DataSets.get('Scored_test_data_CatBoost_1')

    # Check ModelArgs Dict
    x.ModelArgs

    # Check the names of data sets collected
    x.DataSetsNames

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
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
    # Create partitioned data sets
    DataFrames = fe.FE2_AutoDataParition(
      data=data, 
      ArgsList=None, 
      DateColumnName=None, 
      PartitionType='random', 
      Ratios=[0.7,0.2,0.1], 
      ByVariables=None, 
      Sort=False, 
      Processing='datatable', 
      InputFrame='datatable', 
      OutputFrame='datatable')
    
    # Prepare modeling data sets
    ModelData = ml.ML0_GetModelData(
      Processing='xgboost',
      TrainData=DataFrames['TrainData'],
      ValidationData=DataFrames['ValidationData'],
      TestData=DataFrames['TestData'],
      ArgsList=None,
      TargetColumnName='Leads',
      NumericColumnNames=['XREGS1', 'XREGS2', 'XREGS3'],
      CategoricalColumnNames=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'],
      TextColumnNames=None,
      WeightColumnName=None,
      Threads=-1,
      InputFrame='datatable')
    
    # Get args list for algorithm and target type
    ModelArgs = ml.ML0_Parameters(
      Algorithms='XGBoost', 
      TargetType="Regression", 
      TrainMethod="Train")
    
    # Update iterations to run quickly
    ModelArgs['XGBoost']['AlgoArgs']['num_boost_round'] = 50
    
    # Initialize RetroFit
    x = ml.RetroFit(ModelArgs, ModelData, DataFrames)
    
    # Train Model
    x.ML1_Single_Train(Algorithm='XGBoost')
    
    # Score data
    x.ML1_Single_Score(DataName=x.DataSetsNames[2], ModelName=x.ModelListNames[0], Algorithm='XGBoost')
    
    # Scoring data names
    x.DataSetsNames
    
    # Scoring data
    x.DataSets.get('Scored_test_data_XGBoost_1')
    
    # Check ModelArgs Dict
    x.PrintAlgoArgs(Algo='XGBoost')
    
    # List of model names
    x.ModelListNames
    
    # List of model fitted names
    x.FitListNames
    """
    
    # Define __init__
    def __init__(self, ModelArgs, ModelData, DataFrames):
      self.ModelArgs = ModelArgs
      self.ModelArgsNames = [*self.ModelArgs]
      self.Runs = len(self.ModelArgs)
      self.DataFrames = DataFrames
      self.DataSets = ModelData
      self.DataSetsNames = [*self.DataSets]
      self.ModelList = dict()
      self.ModelListNames = []
      self.FitList = dict()
      self.FitListNames = []
      self.EvaluationList = dict()
      self.EvaluationListNames = []
      self.InterpretationList = dict()
      self.InterpretationListNames = []
      self.CompareModelsList = dict()
      self.CompareModelsListNames = []

    
    #################################################
    #################################################
    # Function: Print Algo Args
    #################################################
    #################################################
    def PrintAlgoArgs(self, Algo=None):
      from retrofit import utils
      print(utils.printdict(self.ModelArgs[Algo]['AlgoArgs']))
    
    #################################################
    #################################################
    # Function: Train Model
    #################################################
    #################################################
    def ML1_Single_Train(self, Algorithm=None):
      
      # Check
      if len(self.ModelArgs) == 0:
        return print("ModelArgs is empty")

      # Which Algo
      if not Algorithm is None:
        TempArgs = self.ModelArgs[Algorithm]
        #TempArgs = ModelArgs[Algorithm]
      else:
        TempArgs = self.ModelArgs[[*self.ModelArgs][0]]

      #################################################
      # Ftrl Method
      #################################################
      if TempArgs.get('Algorithms').lower() == 'ftrl':

        # Setup Environment
        import datatable
        from datatable import f
        from datatable.models import Ftrl

        # Define training data and target variable
        TrainData = self.DataSets.get('train_data')
        TargetColumnName = self.DataSets.get('ArgsList').get('TargetColumnName')

        # Initialize model
        Model = Ftrl(**TempArgs.get('AlgoArgs'))
        self.ModelList[f"Ftrl{str(len(self.ModelList) + 1)}"] = Model
        self.ModelListNames.append(f"Ftrl{str(len(self.ModelList))}")

        # Train Model
        self.FitList[f"Ftrl{str(len(self.FitList) + 1)}"] = Model.fit(TrainData[:, f[:].remove(f[TargetColumnName])], TrainData[:, TargetColumnName])
        self.FitListNames.append(f"Ftrl{str(len(self.FitList))}")

      #################################################
      # CatBoost Method
      #################################################
      if TempArgs.get('Algorithms').lower() == 'catboost':

        # Setup Environment
        import catboost
        if TempArgs.get('TargetType').lower() in ['classification', 'multiclass']:
          from catboost import CatBoostClassifier
        else:
          from catboost import CatBoostRegressor

        # Define training data and target variable
        TrainData = self.DataSets.get('train_data')
        ValidationData = self.DataSets.get('validation_data')
        TestData = self.DataSets.get('test_data')
        
        # Initialize model
        if TempArgs.get('TargetType').lower() == 'regression':
          Model = CatBoostRegressor(**TempArgs.get('AlgoArgs'))
        else:
          Model = CatBoostClassifier(**TempArgs.get('AlgoArgs'))
        
        # Store Model
        self.ModelList[f"CatBoost{str(len(self.ModelList) + 1)}"] = Model
        self.ModelListNames.append(f"CatBoost{str(len(self.ModelList))}")

        # Train Model
        self.FitList[f"CatBoost{str(len(self.FitList) + 1)}"] = Model.fit(X=TrainData, eval_set=ValidationData, use_best_model=True)
        self.FitListNames.append(f"CatBoost{str(len(self.FitList))}")

      #################################################
      # XGBoost Method
      #################################################
      if TempArgs.get('Algorithms').lower() == 'xgboost':

        # Setup Environment
        import xgboost as xgb
        from xgboost import train
        
        # Define training data and target variable
        TrainData = self.DataSets.get('train_data')
        ValidationData = self.DataSets.get('validation_data')
        TestData = self.DataSets.get('test_data')
        
        # Initialize model
        Model = xgb.XGBModel(**TempArgs.get('AlgoArgs'))
          
        # Store Model
        self.ModelList[f"XGBoost{str(len(self.ModelList) + 1)}"] = Model
        self.ModelListNames.append(f"XGBoost{str(len(self.ModelList))}")

        # Train Model
        self.FitList[f"XGBoost{str(len(self.FitList) + 1)}"] = xgb.train(params=TempArgs.get('AlgoArgs'), dtrain=TrainData, evals=[(ValidationData, 'Validate'), (TestData, 'Test')], num_boost_round=TempArgs.get('AlgoArgs').get('num_boost_round'), early_stopping_rounds=TempArgs.get('AlgoArgs').get('early_stopping_rounds'))
        self.FitListNames.append(f"XGBoost{str(len(self.FitList))}")
        
      #################################################
      # LightGBM Method
      #################################################
      if TempArgs.get('Algorithms').lower() == 'lightgbm':

        # Setup Environment
        import lightgbm as lgbm
        from lightgbm import LGBMModel
        
        # Define training data and target variable
        TrainData = self.DataSets.get('train_data')
        ValidationData = self.DataSets.get('validation_data')
        TestData = self.DataSets.get('test_data')

        # Initialize model
        Model = LGBMModel(**TempArgs.get('AlgoArgs'))
        
        # Store Model
        self.ModelList[f"LightGBM{str(len(self.ModelList) + 1)}"] = Model
        self.ModelListNames.append(f"LightGBM{str(len(self.ModelList))}")

        # Initialize model
        import copy
        temp_args = copy.deepcopy(TempArgs)
        del temp_args['AlgoArgs']['num_iterations']
        del temp_args['AlgoArgs']['early_stopping_round']
        self.FitList[f"LightGBM{str(len(self.FitList) + 1)}"] = lgbm.train(params=temp_args.get('AlgoArgs'), train_set=TrainData, valid_sets=[ValidationData, TestData], num_boost_round=TempArgs.get('AlgoArgs').get('num_iterations'), early_stopping_rounds=TempArgs.get('AlgoArgs').get('early_stopping_round'))
        self.FitListNames.append(f"LightGBM{str(len(self.FitList))}")

    #################################################
    #################################################
    # Function: Score data 
    #################################################
    #################################################
    def ML1_Single_Score(self, DataName=None, ModelName=None, Algorithm=None, NewData=None):

      # Check
      if len(self.ModelList) == 0:
        return print("No models exist")

      # Which Algo
      if not Algorithm is None:
        TempArgs = self.ModelArgs[Algorithm]
      else:
        TempArgs = self.ModelArgs[[*self.ModelArgs][0]]

      # Setup Environment
      import datatable
      from datatable.models import Ftrl

      #################################################
      # Ftrl Method
      #################################################
      if TempArgs['Algorithms'].lower() == 'ftrl':
        
        # Extract model
        if not ModelName is None:
          Model = self.ModelList.get(ModelName)
        else:
          Model = self.ModelList.get(f"Ftrl_{str(len(self.FitList))}")

        # Grab scoring data
        TargetColumnName = self.DataSets.get('ArgsList')['TargetColumnName']
        if NewData is None:
          score_data = self.DataSets[DataName]
        else:
          score_data = NewData
        
        # Split frames
        if TargetColumnName in score_data.names:
          TargetData = score_data[:, f[TargetColumnName]]
          score_data = score_data[:, f[:].remove(f[TargetColumnName])]
          if NewData is None:
            return Model.predict(score_data)

        # Score Model and append data set name to scoring data
        score_data.cbind(Model.predict(score_data))
        
        # Update prediction column name
        score_data.names = {TargetColumnName: f"Predict_{TargetColumnName}"}
        
        # cbind Target column back to score_data
        score_data.cbind(TargetData)

        # Store data and update names
        self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}"] = score_data
        self.DataSetsNames.append(f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}")

      #################################################
      # CatBoost Method
      #################################################
      if TempArgs['Algorithms'].lower() == 'catboost':

        # Extract Model
        if not ModelName is None:
          Model = self.ModelList.get(ModelName)
        else:
          Model = self.ModelList.get(f"CatBoost_{str(len(self.FitList))}")

        # Grab dataframe data
        TargetColumnName = self.DataSets.get('ArgsList')['TargetColumnName']
        if NewData is None:
          pred_data = self.DataSets[DataName]
          if DataName == 'test_data':
            ScoreData = self.DataFrames.get('TestData')
          elif DataName == 'validation_data':
            ScoreData = self.DataFrames.get('ValidationData')
          elif DataName == 'train_data':
            ScoreData = self.DataFrames.get('TrainData')
        else:
          pred_data = NewData

        # Generate preds and add to datatable frame
        if NewData is None:
          if TempArgs.get('TargetType').lower() == 'regression':
            ScoreData[f"Predict_{TargetColumnName}"] = Model.predict(pred_data, prediction_type = 'RawFormulaVal')
          elif TempArgs.get('TargetType').lower() == 'classification':
            ScoreData[f"Predict_{TargetColumnName}"] = Model.predict(pred_data, prediction_type = 'Probability')
          elif TempArgs.get('TargetType').lower() == 'multiclass':
            ScoreData[f"Predict_{TargetColumnName}"] = Model.predict(pred_data, prediction_type = 'Class')
        else:
          if TempArgs.get('TargetType').lower() == 'regression':
            return Model.predict(pred_data, prediction_type = 'RawFormulaVal')
          elif TempArgs.get('TargetType').lower() == 'classification':
            return Model.predict(pred_data, prediction_type = 'Probability')
          elif TempArgs.get('TargetType').lower() == 'multiclass':
            return Model.predict(pred_data, prediction_type = 'Class')

        # Store data and update names
        self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}"] = ScoreData
        self.DataSetsNames.append(f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}")

      #################################################
      # XGBoost Method
      #################################################
      if TempArgs['Algorithms'].lower() == 'xgboost':

        # Environment
        import xgboost as xgb

        # Extract Model
        if not ModelName is None:
          Model = self.FitList.get(ModelName)
        else:
          Model = self.FitList.get(f"XGBoost_{str(len(self.FitList))}")

        # Grab dataframe data
        TargetColumnName = self.DataSets.get('ArgsList')['TargetColumnName']
        if NewData is None:
          pred_data = self.DataSets[DataName]
          if DataName == 'test_data':
            ScoreData = self.DataFrames.get('TestData')
          elif DataName == 'validation_data':
            ScoreData = self.DataFrames.get('ValidationData')
          elif DataName == 'train_data':
            ScoreData = self.DataFrames.get('TrainData')
        else:
          score_data = NewData
          pred_data = self.DataSets[DataName]

        # Generate preds and add to datatable frame
        if NewData is None:
          ScoreData[f"Predict_{TargetColumnName}"] = Model.predict(
            data = pred_data, 
            output_margin=False, 
            pred_leaf=False, 
            pred_contribs=False, # shap values: creates a matrix output
            approx_contribs=False, 
            pred_interactions=False, 
            validate_features=True, 
            training=False, 
            iteration_range=(0, self.FitList[f"XGBoost{str(len(self.FitList))}"].best_iteration), 
            strict_shape=False)
        else:
          return Model.predict(
            data = pred_data, 
            output_margin=False, 
            pred_leaf=False, 
            pred_contribs=False, # shap values: creates a matrix output
            approx_contribs=False, 
            pred_interactions=False, 
            validate_features=True, 
            training=False, 
            iteration_range=(0, self.FitList[f"XGBoost{str(len(self.FitList))}"].best_iteration), 
            strict_shape=False)
        
        # Store data and update names
        self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}"] = ScoreData
        self.DataSetsNames.append(f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}")
      
      #################################################
      # LightGBM Method
      #################################################
      if TempArgs['Algorithms'].lower() == 'xgboost':
        
        # Environment
        import lightgbm as lgbm
        
        # Extract Model
        if not ModelName is None:
          Model = self.FitList.get(ModelName)
        else:
          Model = self.FitList.get(f"LightGBM{str(len(self.FitList))}")

        # Grab dataframe data
        TargetColumnName = self.DataSets.get('ArgsList')['TargetColumnName']
        if NewData is None:
          pred_data = self.DataSets[DataName]
          if DataName == 'test_data':
            ScoreData = self.DataFrames.get('TestData')
          elif DataName == 'validation_data':
            ScoreData = self.DataFrames.get('ValidationData')
          elif DataName == 'train_data':
            ScoreData = self.DataFrames.get('TrainData')
        else:
          pred_data = NewData

        # Generate preds and add to datatable frame
        if NewData is None:
          ScoreData[f"Predict_{TargetColumnName}"] = Model.predict(
            data = pred_data, 
            output_margin=False, 
            pred_leaf=False, 
            pred_contribs=False, # shap values: creates a matrix output
            approx_contribs=False, 
            pred_interactions=False, 
            validate_features=True, 
            training=False, 
            iteration_range=(0, self.FitList[f"LightGBM{str(len(self.FitList))}"].best_iteration), 
            strict_shape=False)
        else:
          return Model.predict(
            data = pred_data, 
            output_margin=False, 
            pred_leaf=False, 
            pred_contribs=False, # shap values: creates a matrix output
            approx_contribs=False, 
            pred_interactions=False, 
            validate_features=True, 
            training=False, 
            iteration_range=(0, self.FitList[f"LightGBM{str(len(self.FitList))}"].best_iteration), 
            strict_shape=False)
        
        # Store data and update names
        self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}"] = ScoreData
        self.DataSetsNames.append(f"Scored_{DataName}_{Algorithm}_{len(self.FitList)}")
