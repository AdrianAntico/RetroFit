# Module: MachineLearning
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.0
# Last modified : 2021-09-03

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
      # if not CategoricalColumnNames is None:
      #     SD.extend(CategoricalColumnNames)
      # if not TextColumnNames is None:
      #     SD.extend(TextColumnNames)
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
      if not CategoricalColumnNames is None:
        SD.extend(CategoricalColumnNames)
      if not TextColumnNames is None:
        SD.extend(TextColumnNames)
      if not WeightColumnName is None:
        trainweightdata = TrainData['WeightColumnName']
        if not ValidationData is None:
          validationweightdata = ValidationData['WeightColumnName']
        if not TestData is None:
          testweightdata = TestData['WeightColumnName']
      else:
        trainweightdata = None
        validationweightdata = None
        testweightdata = None
        
      # data
      train = TrainData[:, SD]
      if not ValidationData is None:
        validation = ValidationData[:, SD]
      if not TestData is None:
        test = TestData[:, SD]

      # label
      trainlabel = TrainData[:, TargetColumnName]
      if not ValidationData is None:
        validationlabel = ValidationData[:, TargetColumnName]
      if not TestData is None:
        testlabel = TestData[:, TargetColumnName]

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
    Algorithms: Choose from CatBoost, XGBoost, LightGBM, Ftrl
    TargetType: Choose from 'regression', 'classification', 'multiclass'
    TrainMethod: Choose from 'train', 'gridtune'
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

      #############################################
      # Algorithm Selection XGBoost
      #############################################
      if Algo.lower() == 'xgboost':
    
        # Setup Environment
        import xgboost as xgb
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

      #############################################
      # Algorithm Selection LightGBM
      #############################################
      if Algo.lower() == 'lightgbm':
    
        # Setup Environment
        import lightgbm as lgbm
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
    
    Class Initialization
    Model Initialization
    Training
    Grid Tuning
    Scoring
    Model Evaluation
    Model Interpretation
    
    ####################################
    # Functions
    ####################################
    
    FUN_Train()
    FUN_GridTune()
    FUN_Score()
    
    ####################################
    # Example Usage
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
    DataSets = ml.ML0_GetModelData(
      Processing='Ftrl',
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
      Algorithms='Ftrl', 
      TargetType="Regression", 
      TrainMethod="Train")

    # Initialize RetroFit
    x = RetroFit(ModelArgs, DataSets)

    # Train Model
    x.ML1_Single_Train(Algorithm='Ftrl')

    # Score data
    x.ML1_Single_Score(DataName=x.DataSetsNames[2], ModelName=x.ModelListNames[0])

    # Scoring data names
    x.DataSets['Scored_test_data'].names

    # Check ModelArgs Dict
    x.ModelArgs

    # Check the names of data sets collected
    x.DataSetsNames

    # List of model names
    x.ModelListNames

    # List of model fitted names
    x.FitListNames

    # List of comparisons
    x.CompareModelsListNames
    """
  
    # Define __init__
    def __init__(self, ModelArgs, DataSets):
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

    # Train Model
    def ML1_Single_Train(self, Algorithm=None):
      
      # Check
      if len(self.ModelArgs) == 0:
        return print("ModelArgs is empty")

      # Which Algo
      if not Algorithm is None:
        TempArgs = self.ModelArgs[Algorithm]
      else:
        TempArgs = self.ModelArgs[[*self.ModelArgs][0]]

      # Train Ftrl
      if TempArgs.get('Algorithms').lower() == 'ftrl':

        # Setup Environment
        import datatable
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

    # Score data @staticmethod
    def ML1_Single_Score(self, DataName=None, ModelName=None, Algorithm=None):

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

      # Score model
      if TempArgs['Algorithms'].lower() == 'ftrl':
        
        # Extract model
        if not ModelName is None:
          model = self.ModelList.get(ModelName)
        else:
          model = self.ModelList.get(f"Ftrl_{str(len(self.FitList))}")

        # Extract scoring data
        TargetColumnName = self.DataSets.get('ArgsList')['TargetColumnName']
        score_data = self.DataSets[DataName]
        if TargetColumnName in score_data.names:
          TargetData = score_data[:, f[TargetColumnName]]
          score_data = score_data[:, f[:].remove(f[TargetColumnName])]

        # Score model and append data set name to scoring data
        score_data.cbind(model.predict(score_data))
        
        # Update prediction column name
        score_data.names = {TargetColumnName: f"Predict_{TargetColumnName}"}
        
        # cbind Target column back to score_data
        score_data.cbind(TargetData)

        # Store data and update names
        if not 'Scored_' + DataName in score_data.names:
          self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(DataName)+1}"] = score_data
          self.DataSetsNames.append('Scored_' + DataName)
        else:
          self.DataSets[f"Scored_{DataName}_{Algorithm}_{len(DataName)+1}"] = score_data
          self.DataSetsNames.append(f"Scored_{DataName}_{Algorithm}_{len(DataName)+1}")
