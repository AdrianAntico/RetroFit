# Module: MachineLearning
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.2.0
# Last modified : 2025-11-15

import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from numpy import sort
from retrofit import utils as u
import os
from copy import copy
import pandas as pd
import polars as pl
import catboost
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
from xgboost import train, Booster
import lightgbm as lgbm
from lightgbm import LGBMModel
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    multilabel_confusion_matrix,
    top_k_accuracy_score,
    confusion_matrix,
    hamming_loss,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score
) 


class RetroFit:
    """
    Goals:
      Training
      Feature Tuning
      Grid Tuning
      Continued Training
      Scoring
      Model Evaluation
      Model Interpretation

    Functions:
      train
      score
      save
      load
      evaluate
      print_algo_args

    Attributes:
      self.TargetType = "regression"
      self.Algorithm = "catboost"
      self.TargetColumnName = ModelData["ArgsList"]["TargetColumnName"]
      self.NumericColumnNames = ModelData["ArgsList"]["NumericColumnNames"]
      self.CategoricalColumnNames = ModelData["ArgsList"]["CategoricalColumnNames"]
      self.TextColumnNames = ModelData["ArgsList"]["TextColumnNames"]
      self.WeightColumnName = ModelData["ArgsList"]["WeightColumnName"]
      self.ModelArgs = ModelArgs
      self.ModelArgsNames = [*self.ModelArgs]
      self.Runs = len(self.ModelArgs)
      self.DataSets = DataSets
      self.ModelData = ModelData
      self.self.ModelDataNames = [*self.DataSets]
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
    """

    def __init__(
        self,
        Algorithm: str = "catboost",
        TargetType: str = "regression"
    ):
    
        # Model info
        self.Algorithm = Algorithm.lower()
        self.TargetType = TargetType.lower()
    
        # Original data in POLARS
        self.DataFrames = {
            "train": None,
            "validation": None,
            "test": None,
        }
    
        # Optional: store scored versions (Polars)
        self.ScoredData = {}
    
        # Data columns by type
        self.TargetColumnName = None
        self.NumericColumnNames = []
        self.CategoricalColumnNames = []
        self.TextColumnNames = []
        self.WeightColumnName = None
    
        # Model parameters
        self.ModelArgs = None
        self.ModelArgsNames = None
    
        # Algo-specific data objects (Pool / DMatrix / Dataset)
        self.ModelData = None
        self.ModelDataNames = None
    
        # Main model handle (single-run convenience)
        self.Model = None
    
        # Model info (for multiple runs / variants)
        self.ModelList = {}
        self.ModelListNames = []
    
        # Models saved
        self.SavedModels = []
    
        # Models fitted (e.g., booster objects)
        self.FitList = {}
        self.FitListNames = []
    
        # Model evaluations
        self.EvaluationList = {}
        self.EvaluationListNames = []
    
        # Model interpretations
        self.InterpretationList = {}
        self.InterpretationListNames = []
    
        # Model comparisons
        self.CompareModelsList = {}
        self.CompareModelsListNames = []


    #################################################
    # Function: Create Model-Data Objects
    #################################################
    
    # Helper function: normalize input to POLARS
    @staticmethod
    def _normalize_input_df(df):
        """
        Normalize user-supplied dataframes to Polars internally.
        Accepts:
            - None
            - polars.DataFrame
            - pandas.DataFrame (converted to Polars)
        """
        if df is None:
            return None
    
        if isinstance(df, pl.DataFrame):
            return df
    
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    
        raise ValueError("Input must be a polars or pandas DataFrame (or None).")
    
    
    # Helper function: convert to pandas for model boundaries
    @staticmethod
    def _to_pandas(df):
        """
        Convert internal Polars → pandas for building algo-specific data objects.
        Accepts:
            - None
            - polars.DataFrame
            - pandas.DataFrame
        """
        if df is None:
            return None
    
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    
        if isinstance(df, pd.DataFrame):
            return df
    
        raise ValueError("Input must be a polars or pandas DataFrame (or None).")
    
    
    # Helper function: CatBoost
    @staticmethod
    def _process_catboost(
        TrainData=None,
        ValidationData=None,
        TestData=None,
        TargetColumnName=None,
        NumericColumnNames=None,
        CategoricalColumnNames=None,
        TextColumnNames=None,
        WeightColumnName=None,
        Threads=None
    ):
        """
        Build CatBoost Pool objects from pandas DataFrames.
        """
        def create_pool(data, label, cat_features, text_features, weight):
            return Pool(
                data=data,
                label=label,
                cat_features=cat_features,
                text_features=text_features,
                weight=weight,
                thread_count=Threads
            ) if data is not None else None
    
        lists = [NumericColumnNames, CategoricalColumnNames, TextColumnNames]
        cols = [c for group in lists if group for c in group]
    
        train_pool = create_pool(
            data=TrainData[cols],
            label=TrainData[TargetColumnName],
            cat_features=CategoricalColumnNames,
            text_features=TextColumnNames,
            weight=TrainData[WeightColumnName] if WeightColumnName else None
        )
    
        valid_pool = create_pool(
            data=ValidationData[cols] if ValidationData is not None else None,
            label=ValidationData[TargetColumnName] if ValidationData is not None else None,
            cat_features=CategoricalColumnNames,
            text_features=TextColumnNames,
            weight=ValidationData[WeightColumnName] if ValidationData is not None and WeightColumnName else None
        )
    
        test_pool = create_pool(
            data=TestData[cols] if TestData is not None else None,
            label=TestData[TargetColumnName] if TestData is not None else None,
            cat_features=CategoricalColumnNames,
            text_features=TextColumnNames,
            weight=TestData[WeightColumnName] if TestData is not None and WeightColumnName else None
        )
    
        return {"train_data": train_pool, "validation_data": valid_pool, "test_data": test_pool}
    
    
    # Helper function: XGBoost
    @staticmethod
    def _process_xgboost(
        TrainData=None,
        ValidationData=None,
        TestData=None,
        TargetColumnName=None,
        NumericColumnNames=None,
        WeightColumnName=None
    ):
        """
        Build XGBoost DMatrix objects from pandas DataFrames.
        """
        def create_dmatrix(data, label, weight):
            return xgb.DMatrix(data=data, label=label, weight=weight) if data is not None else None
    
        train_dmatrix = create_dmatrix(
            data=TrainData[NumericColumnNames],
            label=TrainData[TargetColumnName],
            weight=TrainData[WeightColumnName] if WeightColumnName else None
        )
    
        valid_dmatrix = create_dmatrix(
            data=ValidationData[NumericColumnNames] if ValidationData is not None else None,
            label=ValidationData[TargetColumnName] if ValidationData is not None else None,
            weight=ValidationData[WeightColumnName] if ValidationData is not None and WeightColumnName else None
        )
    
        test_dmatrix = create_dmatrix(
            data=TestData[NumericColumnNames] if TestData is not None else None,
            label=TestData[TargetColumnName] if TestData is not None else None,
            weight=TestData[WeightColumnName] if TestData is not None and WeightColumnName else None
        )
    
        return {"train_data": train_dmatrix, "validation_data": valid_dmatrix, "test_data": test_dmatrix}
    
    
    # Helper function: LightGBM
    @staticmethod
    def _process_lightgbm(
        TrainData=None,
        ValidationData=None,
        TestData=None,
        TargetColumnName=None,
        NumericColumnNames=None,
        WeightColumnName=None
    ):
        """
        Build LightGBM Dataset objects from pandas DataFrames.
        """
        def create_dataset(data, label, weight):
            return lgbm.Dataset(data=data, label=label, weight=weight) if data is not None else None
    
        train_dataset = create_dataset(
            data=TrainData[NumericColumnNames],
            label=TrainData[TargetColumnName],
            weight=TrainData[WeightColumnName] if WeightColumnName else None
        )
    
        valid_dataset = create_dataset(
            data=ValidationData[NumericColumnNames] if ValidationData is not None else None,
            label=ValidationData[TargetColumnName] if ValidationData is not None else None,
            weight=ValidationData[WeightColumnName] if ValidationData is not None and WeightColumnName else None
        )
    
        test_dataset = create_dataset(
            data=TestData[NumericColumnNames] if TestData is not None else None,
            label=TestData[TargetColumnName] if TestData is not None else None,
            weight=TestData[WeightColumnName] if TestData is not None and WeightColumnName else None
        )
    
        return {"train_data": train_dataset, "validation_data": valid_dataset, "test_data": test_dataset}
    
    
    # Main function
    def create_model_data(
        self,
        TrainData=None,
        ValidationData=None,
        TestData=None,
        TargetColumnName=None,
        NumericColumnNames=None,
        CategoricalColumnNames=None,
        TextColumnNames=None,
        WeightColumnName=None,
        Threads=-1
    ):
        """
        Create modeling objects for specific algorithms (CatBoost, XGBoost, LightGBM).
    
        Parameters:
            TrainData, ValidationData, TestData: pandas or polars DataFrames.
            TargetColumnName: Target column name.
            NumericColumnNames, CategoricalColumnNames, TextColumnNames: Lists of column names.
            WeightColumnName: Column name for sample weights.
            Threads: Number of threads to utilize.
    
        Side effects:
            - Stores POLARS originals in self.DataFrames["train"/"validation"/"test"]
            - Stores algo-specific objects in self.ModelData
            - Initializes self.ModelArgs via create_model_parameters()
        """
    
        # 1) Normalize all inputs to POLARS internally
        TrainData = self._normalize_input_df(TrainData)
        ValidationData = self._normalize_input_df(ValidationData)
        TestData = self._normalize_input_df(TestData)
    
        # 2) Store metadata / column info
        self.TargetColumnName = TargetColumnName
        self.NumericColumnNames = NumericColumnNames or []
        self.CategoricalColumnNames = CategoricalColumnNames or []
        self.TextColumnNames = TextColumnNames or []
        self.WeightColumnName = WeightColumnName
    
        # 3) Store original POLARS frames for later scoring/evaluation
        self.DataFrames["train"] = TrainData
        self.DataFrames["validation"] = ValidationData
        self.DataFrames["test"] = TestData
    
        # 4) Convert to pandas for model objects
        train_pd = self._to_pandas(TrainData)
        valid_pd = self._to_pandas(ValidationData)
        test_pd = self._to_pandas(TestData)
    
        # 5) Select processing pipeline
        if self.Algorithm == 'catboost':
            self.ModelData = self._process_catboost(
                TrainData=train_pd,
                ValidationData=valid_pd,
                TestData=test_pd,
                TargetColumnName=TargetColumnName,
                NumericColumnNames=self.NumericColumnNames,
                CategoricalColumnNames=self.CategoricalColumnNames,
                TextColumnNames=self.TextColumnNames,
                WeightColumnName=WeightColumnName,
                Threads=Threads
            )
            self.ModelDataNames = [*self.ModelData]
    
        elif self.Algorithm == 'xgboost':
            if not self.NumericColumnNames:
                raise ValueError("NumericColumnNames must be provided for XGBoost.")
            self.ModelData = self._process_xgboost(
                TrainData=train_pd,
                ValidationData=valid_pd,
                TestData=test_pd,
                TargetColumnName=TargetColumnName,
                NumericColumnNames=self.NumericColumnNames,
                WeightColumnName=WeightColumnName
            )
            self.ModelDataNames = [*self.ModelData]
    
        elif self.Algorithm == 'lightgbm':
            if not self.NumericColumnNames:
                raise ValueError("NumericColumnNames must be provided for LightGBM.")
            self.ModelData = self._process_lightgbm(
                TrainData=train_pd,
                ValidationData=valid_pd,
                TestData=test_pd,
                TargetColumnName=TargetColumnName,
                NumericColumnNames=self.NumericColumnNames,
                WeightColumnName=WeightColumnName
            )
            self.ModelDataNames = [*self.ModelData]
    
        else:
            raise ValueError("Unsupported processing type. Choose from 'catboost', 'xgboost', or 'lightgbm'.")
    
        # 6) Initialize base model parameters for this algorithm/target type
        self.create_model_parameters()

    
    #################################################
    # Function: Create Algo-Specific Args
    #################################################
    def create_model_parameters(self):
        """
        # Goal
        Return an ArgsList appropriate for the algorithm selection, target type, and training method
        """

        # Args Check
        if self.Algorithm is None:
          raise Exception('Algorithm cannot be None')
    
        if self.TargetType is None:
          raise Exception('TargetType cannot be None')

        # CatBoost
        if self.Algorithm == 'catboost':
    
          # Initialize AlgoArgs
          AlgoArgs = dict()
    
          ###############################
          # TargetType Parameters
          ###############################
          if self.TargetType == 'classification':
            AlgoArgs['loss_function'] = 'Logloss'
            AlgoArgs['eval_metric'] = 'Logloss'
            AlgoArgs['auto_class_weights'] = 'Balanced'
          elif self.TargetType == 'multiclass':
            AlgoArgs['classes_count'] = 3
            AlgoArgs['loss_function'] = 'MultiClassOneVsAll'
            AlgoArgs['eval_metric'] = 'MultiClassOneVsAll'
          elif self.TargetType == 'regression':
            AlgoArgs['loss_function'] = 'RMSE'
            AlgoArgs['eval_metric'] = 'RMSE'
    
          ###############################
          # Parameters
          ###############################
          AlgoArgs['train_dir'] = os.getcwd()
          AlgoArgs['task_type'] = 'CPU'
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
          AlgoArgs['iterations'] = 1000
          AlgoArgs['depth'] = 6
          AlgoArgs['langevin'] = True
          AlgoArgs['diffusion_temperature'] = 10000
          AlgoArgs['grow_policy'] = 'SymmetricTree'
          AlgoArgs['model_size_reg'] = 0.5
          
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
    
        # XGBoost
        if self.Algorithm == 'xgboost':
      
          # Setup Environment
          
          AlgoArgs = dict()
          
          # Performance Params
          AlgoArgs['nthread'] = os.cpu_count()
          AlgoArgs['predictor'] = 'auto'
          AlgoArgs['single_precision_histogram'] = False
          AlgoArgs['early_stopping_rounds'] = 50
          
          # Training Params
          AlgoArgs['tree_method'] = 'hist'
          AlgoArgs['max_bin'] = 256
          
          ###############################
          # Gridable Parameters
          ###############################
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
    
          # GPU Dependent
          if AlgoArgs['tree_method'] == 'gpu_hist':
            AlgoArgs['sampling_method'] = 'uniform'
    
          # Target Dependent Args
          if self.TargetType == 'classification':
            AlgoArgs['objective'] = 'binary:logistic'
            AlgoArgs['eval_metric'] = 'auc'
          elif self.TargetType == 'regression':
            AlgoArgs['objective'] = 'reg:squarederror'
            AlgoArgs['eval_metric'] = 'rmse'
          elif self.TargetType == 'multiclass':
            AlgoArgs['objective'] = 'multi:softprob'
            AlgoArgs['eval_metric'] = 'mlogloss'
    
        # LightGBM
        if self.Algorithm == 'lightgbm':
      
          # Setup Environment
          AlgoArgs = dict()
          
          # Target Dependent Args
          if self.TargetType == 'classification':
            AlgoArgs['objective'] = 'binary'
            AlgoArgs['metric'] = 'auc'
          elif self.TargetType == 'regression':
            AlgoArgs['objective'] = 'regression'
            AlgoArgs['metric'] = 'rmse'
          elif self.TargetType == 'multiclass':
            AlgoArgs['objective'] = 'multiclassova'
            AlgoArgs['metric'] = 'multi_logloss'
    
          # Tuning Args
          AlgoArgs['num_iterations'] = 1000
          AlgoArgs['learning_rate'] = None
          AlgoArgs['num_leaves'] = 31
          AlgoArgs['bagging_freq'] = 0
          AlgoArgs['bagging_fraction'] = 1.0
          AlgoArgs['feature_fraction'] = 1.0
          AlgoArgs['feature_fraction_bynode'] = 1.0
          AlgoArgs['max_delta_step'] = 0.0
          
          # Args
          AlgoArgs['task'] = 'train'
          AlgoArgs['device_type'] = 'CPU'
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

        # Store Model Parameters
        self.ModelArgs = AlgoArgs
        self.ModelArgsNames = [*self.ModelArgs]


    #################################################
    # Function: Update Algo-Specific Args
    #################################################
    def update_model_parameters(
        self,
        allow_new: bool = False,
        **kwargs
    ):
        """
        Update existing model parameters in self.ModelArgs.
    
        Example:
            model.update_model_parameters(iterations=2000, depth=8)
    
        Parameters
        ----------
        allow_new : bool
            If False (default), raise if a parameter name does not already
            exist in self.ModelArgs. If True, new keys are allowed.
        **kwargs :
            Parameter names and values to update.
        """
        if self.ModelArgs is None:
            raise RuntimeError(
                "ModelArgs is None. Call create_model_data() (which calls "
                "create_model_parameters()) before updating parameters."
            )
    
        for key, value in kwargs.items():
            if not allow_new and key not in self.ModelArgs:
                raise KeyError(
                    f"Parameter '{key}' is not in ModelArgs for algorithm "
                    f"'{self.Algorithm}'. Existing keys: {list(self.ModelArgs.keys())}"
                )
            self.ModelArgs[key] = value


    #################################################
    # Function: Print Algo Args
    #################################################
    def print_algo_args(self):
        print(u.print_dict(self.ModelArgs))


    #################################################
    # Function: Train Model
    #################################################
    def train(self):
        """
        Train a model based on self.Algorithm and self.TargetType using self.ModelData and self.ModelArgs.
    
        Uses:
            - train_data      → for fitting
            - validation_data → for eval / early stopping (if present)
            - test_data       → NOT used here; reserved for evaluate()
    
        Populates:
            - self.Model           (main trained model / booster)
            - self.ModelList       (historical models)
            - self.ModelListNames
            - self.FitList         (same objects as Model, for now)
            - self.FitListNames
        """
    
        # Basic checks
        if self.ModelData is None:
            raise RuntimeError("ModelData is None. Call create_model_data() before train().")
    
        if not self.ModelArgs:
            raise RuntimeError("self.ModelArgs is empty. Call create_model_parameters() before train().")
    
        if self.Algorithm is None:
            raise RuntimeError("self.Algorithm is None. It must be 'catboost', 'xgboost', or 'lightgbm'.")
    
        #################################################
        # CatBoost Method
        #################################################
        if self.Algorithm == 'catboost':
    
            train_pool = self.ModelData["train_data"]
            valid_pool = self.ModelData.get("validation_data")
            # test_pool exists but is intentionally NOT used here:
            # test_pool = self.ModelData.get("test_data")
    
            # Initialize model
            if self.TargetType == "regression":
                model = CatBoostRegressor(**self.ModelArgs)
            elif self.TargetType == "classification":
                model = CatBoostClassifier(**self.ModelArgs)
            elif self.TargetType == "multiclass":
                model = CatBoostClassifier(**self.ModelArgs)
            else:
                raise ValueError(f"Unsupported TargetType for CatBoost: {self.TargetType}")
    
            # Fit model (validation optional)
            if valid_pool is not None:
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    use_best_model=True
                )
            else:
                model.fit(train_pool)
    
            # Store main handle
            self.Model = model
    
            # Track in model lists
            name = f"CatBoost{len(self.ModelList) + 1}"
            self.ModelList[name] = model
            self.ModelListNames.append(name)
    
            self.FitList[name] = model
            self.FitListNames.append(name)
    
            return model  # optional convenience
    
        #################################################
        # XGBoost Method
        #################################################
        if self.Algorithm == 'xgboost':
    
            dtrain = self.ModelData["train_data"]
            dvalid = self.ModelData.get("validation_data")
            # dtest exists but is intentionally NOT used here:
            # dtest  = self.ModelData.get("test_data")
    
            # For multiclass, require num_class to be set in ModelArgs
            if self.TargetType == "multiclass" and "num_class" not in self.ModelArgs:
                raise ValueError(
                    "For multiclass XGBoost, 'num_class' must be present in self.ModelArgs."
                )
    
            # Build evaluation list: ONLY train + validation
            evals = []
            if dtrain is not None:
                evals.append((dtrain, "train"))
            if dvalid is not None:
                evals.append((dvalid, "validation"))
    
            num_boost_round = self.ModelArgs.get("num_boost_round", 100)
            early_stopping_rounds = self.ModelArgs.get("early_stopping_rounds", None)
    
            booster = xgb.train(
                params=self.ModelArgs,
                dtrain=dtrain,
                evals=evals if evals else None,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )
    
            # Store main handle
            self.Model = booster
    
            # Track in model lists
            name = f"XGBoost{len(self.ModelList) + 1}"
            self.ModelList[name] = booster
            self.ModelListNames.append(name)
    
            self.FitList[name] = booster
            self.FitListNames.append(name)
    
            return booster
    
        #################################################
        # LightGBM Method
        #################################################
        if self.Algorithm == 'lightgbm':
    
            train_set = self.ModelData["train_data"]
            valid_set = self.ModelData.get("validation_data")
            # test_set exists but is intentionally NOT used here:
            # test_set  = self.ModelData.get("test_data")
    
            # Build valid_sets list: ONLY validation
            valid_sets = [valid_set] if valid_set is not None else None
    
            num_boost_round = self.ModelArgs.get("num_iterations", 100)
            early_stopping_round = self.ModelArgs.get("early_stopping_round", None)
    
            booster = lgbm.train(
                params=self.ModelArgs,
                train_set=train_set,
                valid_sets=valid_sets,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_round
            )
    
            # Store main handle
            self.Model = booster
    
            # Track in model lists
            name = f"LightGBM{len(self.ModelList) + 1}"
            self.ModelList[name] = booster
            self.ModelListNames.append(name)
    
            self.FitList[name] = booster
            self.FitListNames.append(name)
    
            return booster
    
        # If we reach here, algo was not recognized
        raise ValueError(f"Unsupported Algorithm: {self.Algorithm}")


    #################################################
    # Function: Score data 
    #################################################
    
    def _score_catboost(self, model, df_pl: pl.DataFrame, feature_cols, internal_name: str | None):
        """
        Score a Polars DataFrame with a CatBoost model.
        Uses pre-built Pools for train/validation/test when available.
        """
        import numpy as np
    
        # Use existing Pools for internal splits if available
        pool = None
        if internal_name == "train":
            pool = self.ModelData.get("train_data")
        elif internal_name == "validation":
            pool = self.ModelData.get("validation_data")
        elif internal_name == "test":
            pool = self.ModelData.get("test_data")
    
        # If no pool, build one from df_pl
        if pool is None:
            df_pd = self._to_pandas(df_pl)
            data_pd = df_pd[feature_cols] if feature_cols else df_pd
            pool = Pool(
                data=data_pd,
                label=None,
                cat_features=self.CategoricalColumnNames,
                text_features=self.TextColumnNames,
                thread_count=self.ModelArgs.get("thread_count", -1)
            )
    
        # Prediction logic by TargetType
        if self.TargetType == "regression":
            preds = model.predict(pool, prediction_type="RawFormulaVal")
            preds = np.asarray(preds).ravel()
            col_name = f"Predict_{self.TargetColumnName or 'target'}"
            return df_pl.with_columns(
                pl.Series(col_name, preds)
            )
    
        # For classification / multiclass, we use probabilities
        preds = model.predict(pool, prediction_type="Probability")
        preds = np.asarray(preds)
    
        if self.TargetType == "classification":
            # Binary: CatBoost usually returns Nx2
            if preds.ndim == 1:
                p1 = preds
            else:
                p1 = preds[:, 1]
            p0 = 1.0 - p1
            return df_pl.with_columns([
                pl.Series("p1", p1),
                pl.Series("p0", p0),
            ])
    
        if self.TargetType == "multiclass":
            # preds shape: (N, num_classes)
            n_classes = preds.shape[1]
            cols = [
                pl.Series(f"class_{i}", preds[:, i])
                for i in range(n_classes)
            ]
            return df_pl.with_columns(cols)
    
        raise ValueError(f"Unsupported TargetType for CatBoost scoring: {self.TargetType}")

    def _score_xgboost(self, model, df_pl: pl.DataFrame, internal_name: str | None):
        """
        Score a Polars DataFrame with an XGBoost Booster.
        Uses self.NumericColumnNames for features.
        """
        import numpy as np
    
        if not self.NumericColumnNames:
            raise ValueError("NumericColumnNames must be set for XGBoost scoring.")
    
        # Prepare features
        df_pd = self._to_pandas(df_pl)
        X = df_pd[self.NumericColumnNames]
        dmat = xgb.DMatrix(X)
    
        # Predict
        preds = model.predict(dmat)
        preds = np.asarray(preds)
    
        # Regression
        if self.TargetType == "regression":
            col_name = f"Predict_{self.TargetColumnName or 'target'}"
            return df_pl.with_columns(
                pl.Series(col_name, preds.ravel())
            )
    
        # Binary classification
        if self.TargetType == "classification":
            # XGBoost binary: often returns prob of class 1 as 1D array
            if preds.ndim == 1:
                p1 = preds
            else:
                # In case it returns Nx1
                p1 = preds[:, 0]
            p0 = 1.0 - p1
            return df_pl.with_columns([
                pl.Series("p1", p1),
                pl.Series("p0", p0),
            ])
    
        # Multiclass classification
        if self.TargetType == "multiclass":
            # preds: (N, num_classes)
            n_classes = preds.shape[1]
            cols = [
                pl.Series(f"class_{i}", preds[:, i])
                for i in range(n_classes)
            ]
            return df_pl.with_columns(cols)
    
        raise ValueError(f"Unsupported TargetType for XGBoost scoring: {self.TargetType}")

    def _score_lightgbm(self, model, df_pl: pl.DataFrame, internal_name: str | None):
        """
        Score a Polars DataFrame with a LightGBM Booster.
        Uses self.NumericColumnNames for features.
        """
        import numpy as np
    
        if not self.NumericColumnNames:
            raise ValueError("NumericColumnNames must be set for LightGBM scoring.")
    
        df_pd = self._to_pandas(df_pl)
        X = df_pd[self.NumericColumnNames]
    
        preds = model.predict(X)
        preds = np.asarray(preds)
    
        # Regression
        if self.TargetType == "regression":
            col_name = f"Predict_{self.TargetColumnName or 'target'}"
            return df_pl.with_columns(
                pl.Series(col_name, preds.ravel())
            )
    
        # Binary classification (preds: prob of class 1)
        if self.TargetType == "classification":
            if preds.ndim == 1:
                p1 = preds
            else:
                p1 = preds[:, 0]
            p0 = 1.0 - p1
            return df_pl.with_columns([
                pl.Series("p1", p1),
                pl.Series("p0", p0),
            ])
    
        # Multiclass classification
        if self.TargetType == "multiclass":
            # preds: (N, num_classes)
            n_classes = preds.shape[1]
            cols = [
                pl.Series(f"class_{i}", preds[:, i])
                for i in range(n_classes)
            ]
            return df_pl.with_columns(cols)
    
        raise ValueError(f"Unsupported TargetType for LightGBM scoring: {self.TargetType}")

    def _score_one(
        self,
        df_pl: pl.DataFrame,
        internal_name: str | None,
        model,
        store: bool,
    ):
        """
        Internal helper to score a single Polars DataFrame with the current algorithm
        and optionally store it in self.ScoredData[internal_name].
        """
    
        feature_cols = (self.NumericColumnNames or []) + \
                       (self.CategoricalColumnNames or []) + \
                       (self.TextColumnNames or [])
    
        if self.Algorithm == "catboost":
            scored = self._score_catboost(model, df_pl, feature_cols, internal_name)
        elif self.Algorithm == "xgboost":
            scored = self._score_xgboost(model, df_pl, internal_name)
        elif self.Algorithm == "lightgbm":
            scored = self._score_lightgbm(model, df_pl, internal_name)
        else:
            raise ValueError(f"Unsupported Algorithm in score(): {self.Algorithm}")
    
        # Only store if this is an internal split and store=True
        if store and internal_name is not None:
            self.ScoredData[internal_name] = scored
    
        return scored

    def score(
        self,
        DataName: str | None = None,
        NewData=None,
        ModelName: str | None = None,
        store: bool = True,
        return_results: bool = False,
    ):
        """
        Score data with the trained model.
    
        Behavior
        --------
        - NewData is provided:
            * Score ONLY NewData.
            * Do NOT store in self.ScoredData.
            * Always return the scored Polars DataFrame.
    
        - NewData is None and DataName is None:
            * Score ALL available internal splits: 'train', 'validation', 'test'.
            * Store scored frames in self.ScoredData[split] if store=True.
            * If return_results=True, return dict {split: pl.DataFrame}, else return None.
    
        - NewData is None and DataName is one of {'train','validation','test'}:
            * Score ONLY that split.
            * Store scored frame in self.ScoredData[DataName] if store=True.
            * If return_results=True, return the scored pl.DataFrame, else return None.
        """
    
        # 1) Check model
        if self.Model is None and not self.ModelList:
            raise RuntimeError("No trained models found. Call train() before score().")
    
        # 2) Choose model
        if ModelName is not None:
            model = self.ModelList.get(ModelName) or self.FitList.get(ModelName)
            if model is None:
                raise KeyError(f"Model '{ModelName}' not found in ModelList or FitList.")
        else:
            if self.Model is None:
                raise RuntimeError("self.Model is None and no ModelName provided.")
            model = self.Model
    
        # 3) NewData path → always return, never store
        if NewData is not None:
            df_pl = self._normalize_input_df(NewData)
            scored = self._score_one(
                df_pl=df_pl,
                internal_name=None,   # external data → no internal key
                model=model,
                store=False           # explicitly do not store
            )
            return scored  # ignore return_results in this path
    
        # 4) No NewData and no DataName → score ALL internal splits
        if DataName is None:
            out = {}
            for split in ("train", "validation", "test"):
                df_pl = self.DataFrames.get(split)
                if df_pl is None:
                    continue
    
                scored_split = self._score_one(
                    df_pl=df_pl,
                    internal_name=split,
                    model=model,
                    store=store,
                )
                if return_results:
                    out[split] = scored_split
    
            if (
                self.DataFrames.get("train") is None
                and self.DataFrames.get("validation") is None
                and self.DataFrames.get("test") is None
            ):
                raise RuntimeError(
                    "No internal DataFrames found to score. "
                    "Did you call create_model_data()?"
                )

    
            # return dict if requested, else None
            return out if return_results else None
    
        # 5) Single internal split
        if DataName not in ("train", "validation", "test"):
            raise ValueError("DataName must be one of: 'train', 'validation', 'test'.")
    
        df_pl = self.DataFrames.get(DataName)
        if df_pl is None:
            raise ValueError(f"self.DataFrames['{DataName}'] is None; did you call create_model_data()?")
    
        scored = self._score_one(
            df_pl=df_pl,
            internal_name=DataName,
            model=model,
            store=store,
        )
    
        return scored if return_results else None


    #################################################
    # Function: Evaluation
    #################################################
    def evaluate(
        self,
        DataName: str | None = None,
        df=None,
        FitName: str | None = None,
        ByVariables=None,
        CostDict: dict = dict(tpcost=0.0, fpcost=1.0, fncost=1.0, tncost=0.0),
    ):
        """
        Evaluate model performance on scored data.
    
        Parameters
        ----------
        DataName : {"train","validation","test"} or None
            Name of internally scored dataset in self.ScoredData.
            Ignored if `df` is provided.
        df : polars.DataFrame or pandas.DataFrame or None
            Explicit scored data to evaluate. Must contain target column and
            prediction columns created by score().
        FitName : str or None
            Label for the model/run. Used in output and EvaluationList key.
        ByVariables : str or list[str] or None
            Column(s) to group by. If provided, metrics are computed per group.
        CostDict : dict
            Cost matrix for classification (tpcost, fpcost, fncost, tncost).
    
        Returns
        -------
        For regression:
            pl.DataFrame with one row per group (or single row if no grouping).
    
        For binary classification:
            pl.DataFrame with one row per (group, threshold).
        """
    
        # -------------------------------
        # 0) Normalize ByVariables
        # -------------------------------
        if ByVariables is None:
            by_cols: list[str] = []
        elif isinstance(ByVariables, str):
            by_cols = [ByVariables]
        elif isinstance(ByVariables, (list, tuple)):
            by_cols = list(ByVariables)
        else:
            raise TypeError("ByVariables must be None, a string, or a list/tuple of strings.")
    
        # -------------------------------
        # 1) Resolve scored data (Polars)
        # -------------------------------
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to evaluate().")
    
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )
    
        target = self.TargetColumnName
        if target is None:
            raise RuntimeError("self.TargetColumnName is None — did you call create_model_data()?")
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in scored data.")
    
        # Helper: iterate groups (or single global "group")
        if by_cols:
            groups_iter = df_pl.group_by(by_cols, maintain_order=True)
        else:
            # Fake single group: no grouping columns, just the whole df
            groups_iter = [({} , df_pl)]
    
        # timestamp + model name
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = FitName or "RetroFitModel"
        group_label = ",".join(by_cols) if by_cols else "GLOBAL"
    
        # -------------------------------
        # 2) Regression
        # -------------------------------
        if self.TargetType.lower() == "regression":
            pred_col = f"Predict_{target}"
            if pred_col not in df_pl.columns:
                raise ValueError(
                    f"'{pred_col}' not found — did you run score() (regression)?"
                )
    
            rows = []
    
            for key_vals, gdf in groups_iter:
                # key_vals is either dict-like or scalar depending on Polars version
                if isinstance(key_vals, dict):
                    key_dict = key_vals
                elif not by_cols:  # global case
                    key_dict = {}
                else:
                    # older-style (tuple or scalar)
                    if not isinstance(key_vals, tuple):
                        key_vals = (key_vals,)
                    key_dict = dict(zip(by_cols, key_vals))
    
                y_true = gdf[target].to_numpy()
                y_pred = gdf[pred_col].to_numpy()
    
                if y_true.size == 0:
                    # skip empty group
                    continue
    
                # MSLE positivity check
                msle_ok = (y_true > 0).all() and (y_pred > 0).all()
    
                row = {
                    "ModelName": model_name,
                    "CreateTime": create_time,
                    "GroupingVars": group_label,
                    "n_obs": int(y_true.size),
                    "explained_variance": explained_variance_score(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "median_ae": median_absolute_error(y_true, y_pred),
                    "mape": mean_absolute_percentage_error(y_true, y_pred),
                    "mse": mean_squared_error(y_true, y_pred),
                    "max_error": max_error(y_true, y_pred),
                    "msle": mean_squared_log_error(y_true, y_pred) if msle_ok else -1.0,
                }
    
                # attach group key columns
                row.update(key_dict)
                rows.append(row)
    
            out = pl.DataFrame(rows) if rows else pl.DataFrame([])
    
            key = FitName or f"{self.Algorithm}_regression_{DataName or 'data'}"
            self.EvaluationList[key] = out
            self.EvaluationListNames.append(key)
    
            return out
    
        # -------------------------------
        # 3) Binary Classification
        # -------------------------------
        if self.TargetType.lower() == "classification":
            if "p1" not in df_pl.columns:
                raise ValueError("Missing 'p1' probability column for binary classification.")
    
            tpc = CostDict.get("tpcost", 0.0)
            fpc = CostDict.get("fpcost", 1.0)
            fnc = CostDict.get("fncost", 1.0)
            tnc = CostDict.get("tncost", 0.0)
    
            thresholds = np.linspace(0.0, 1.0, 101)
            all_rows = []
    
            for key_vals, gdf in groups_iter:
                if isinstance(key_vals, dict):
                    key_dict = key_vals
                elif not by_cols:
                    key_dict = {}
                else:
                    if not isinstance(key_vals, tuple):
                        key_vals = (key_vals,)
                    key_dict = dict(zip(by_cols, key_vals))
    
                y_true = gdf[target].to_numpy()
                p1 = gdf["p1"].to_numpy()
    
                N1 = len(y_true)
                if N1 == 0:
                    continue
    
                P1 = np.sum(y_true == 1)
                N0 = N1 - P1
    
                for thr in thresholds:
                    y_pred = (p1 >= thr).astype(int)
    
                    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
                    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
                    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
                    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    
                    Accuracy = (TP + TN) / N1 if N1 else -1.0
                    TPR = TP / P1 if P1 else -1.0
                    TNR = TN / N0 if N0 else -1.0
                    FNR = FN / P1 if P1 else -1.0
                    FPR = FP / N1 if N1 else -1.0
    
                    denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
                    if denom > 0:
                        MCC = (TP * TN - FP * FN) / np.sqrt(denom)
                    else:
                        MCC = -1.0
    
                    if (TP + FP + FN) > 0:
                        F1 = 2 * TP / (2 * TP + FP + FN)
                        F2 = 3 * TP / (2 * TP + FP + FN)
                        F05 = 1.5 * TP / (0.5 * TP + FP + FN)
                    else:
                        F1 = F2 = F05 = -1.0
    
                    PPV = TP / (TP + FP) if (TP + FP) else -1.0
                    NPV = TN / (TN + FN) if (TN + FN) else -1.0
                    Threat = TP / (TP + FP + FN) if (TP + FP + FN) else -1.0
    
                    if (TPR == -1.0) or (FPR == -1.0) or (N1 == 0):
                        Utility = -1.0
                    else:
                        Utility = (
                            (P1 / N1) * (tpc * TPR + fpcost * (1 - TPR))
                            + (1 - P1 / N1) * (fnc * FPR + tnc * (1 - FPR))
                        )
    
                    row = {
                        "ModelName": model_name,
                        "CreateTime": create_time,
                        "GroupingVars": group_label,
                        "Threshold": thr,
                        "n_obs": N1,
                        "P": int(P1),
                        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                        "Accuracy": Accuracy,
                        "TPR": TPR, "TNR": TNR, "FNR": FNR, "FPR": FPR,
                        "F1": F1, "F2": F2, "F0_5": F05,
                        "PPV": PPV, "NPV": NPV,
                        "ThreatScore": Threat,
                        "MCC": MCC,
                        "Utility": Utility,
                    }
    
                    # attach group columns
                    row.update(key_dict)
                    all_rows.append(row)
    
            out = pl.DataFrame(all_rows) if all_rows else pl.DataFrame([])
    
            key = FitName or f"{self.Algorithm}_classification_{DataName or 'data'}"
            self.EvaluationList[key] = out
            self.EvaluationListNames.append(key)
    
            return out
    
        # -------------------------------
        # 4) Multiclass placeholder
        # -------------------------------
        raise NotImplementedError("Multiclass evaluation not yet implemented.")


    #################################################
    # Function: Save / Load entire RetroFit object
    #################################################
    def save_retrofit(self, path):
        """
        Save the entire RetroFit object (including models, args,
        scored data, etc.) to disk via pickle.
        
        Parameters
        ----------
        path : str or Path
            File path to save to, e.g. "models/my_project_retrofit.pkl"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_retrofit(cls, path):
        """
        Load a RetroFit object that was saved with `save_retrofit`.
        
        Parameters
        ----------
        path : str or Path
            File path of the saved pickle.
        
        Returns
        -------
        RetroFit
        """
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {path} is a {type(obj)} not {cls.__name__}"
            )
        return obj
