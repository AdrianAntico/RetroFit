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
from QuickEcharts import Charts
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
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

    # Class attributes
    def __init__(
        self,
        Algorithm: str = "catboost",
        TargetType: str = "regression",
        GPU: bool = False,
    ):
    
        # Model info
        self.Algorithm = Algorithm.lower()
        self.TargetType = TargetType.lower()
        self.GPU = bool(GPU)
    
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

        # Transformations
        self.TargetTransform: str | None = None
        self.TargetTransformParams: dict = {}

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

        # Model importance (single-feature importance)
        self.ImportanceList = {}
        self.ImportanceListNames = []

        # Model interaction importance (e.g., CatBoost pairwise)
        self.InteractionImportanceList = {}
        self.InteractionImportanceListNames = []

        # Calibration Table Storage
        self.CalibrationList = {}
        self.CalibrationListNames = []

        # Label encoding (for classification / multiclass)
        self.LabelMapping = None
        self.LabelMappingInverse = None


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

    # Target Transformations (regression)
    def set_target_transform(self, transform: str | None):
        if transform is None:
            self.TargetTransform = None
            self.TargetTransformParams = {}
            return
    
        t = transform.lower()
        if t not in ("none", "log", "sqrt", "standardize"):
            raise ValueError(
                "Unsupported TargetTransform. "
                "Choose from: None, 'log', 'sqrt', 'standardize'."
            )
    
        self.TargetTransform = t
        self.TargetTransformParams = {}

    # Prepare transformation
    def _prepare_target_transform_params(self, train_df: pl.DataFrame):
        """
        Compute and store any parameters needed for the target transform
        (e.g. shift for 'log', mean/std for 'standardize') using TRAIN data only.
        """
        t = self._normalize_target_transform()
    
        # No transform
        if t == "none":
            self.TargetTransformParams = {}
            return
    
        if self.TargetColumnName is None:
            raise RuntimeError("TargetColumnName is None; cannot prepare target transform params.")
    
        col = self.TargetColumnName
    
        if t == "log":
            # Adaptive shift: only if there are zero or negative values
            stats = train_df.select(pl.col(col).min().alias("min_val")).to_dicts()[0]
            min_y = float(stats["min_val"])
            eps = 1e-6
    
            if min_y <= 0:
                shift = -min_y + eps   # just enough to push min slightly > 0
            else:
                shift = 0.0
    
            self.TargetTransformParams = {"shift": shift}
    
        elif t == "sqrt":
            # Strict: require >= 0
            stats = train_df.select(pl.col(col).min().alias("min_val")).to_dicts()[0]
            min_y = float(stats["min_val"])
            if min_y < 0:
                raise ValueError(
                    f"TargetTransform='sqrt' requires all target values >= 0. "
                    f"Found minimum {min_y}."
                )
            self.TargetTransformParams = {}
    
        elif t == "standardize":
            stats = train_df.select(
                [
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                ]
            ).to_dicts()[0]
    
            if stats["std"] is None or stats["std"] == 0 or np.isnan(stats["std"]):
                raise ValueError(
                    "Target standard deviation is zero or NaN; cannot apply standardize transform."
                )
    
            self.TargetTransformParams = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            }
    
        else:
            raise ValueError(f"Unsupported TargetTransform '{t}'.")

    # Build transform expression
    def _target_transform_expr(self, col_name: str) -> pl.Expr:
        """
        Build a Polars expression that applies the target transform
        to the given column. Used in create_model_data().
        """
        t = self._normalize_target_transform()
    
        if t == "none":
            return pl.col(col_name)
    
        if t == "log":
            shift = float(self.TargetTransformParams.get("shift", 0.0))
            # If shift == 0, this is just log(y)
            return (pl.col(col_name) + shift).log()
    
        if t == "sqrt":
            return pl.col(col_name).sqrt()
    
        if t == "standardize":
            mean = float(self.TargetTransformParams["mean"])
            std = float(self.TargetTransformParams["std"])
            return (pl.col(col_name) - mean) / std
    
        raise ValueError(f"Unsupported TargetTransform '{t}'.")

    # Apply transformation
    def _apply_target_transform_to_internal_frames(self):
        """
        Apply the target transform (if any) to the target column
        in self.DataFrames['train'/'validation'/'test'] using Polars.
    
        Only used for regression.
        """
        t = self._normalize_target_transform()
        if t == "none":
            return
    
        if self.TargetColumnName is None:
            raise RuntimeError("TargetColumnName is None; cannot apply target transform.")
    
        col = self.TargetColumnName
    
        for split in ("train", "validation", "test"):
            df_pl = self.DataFrames.get(split)
            if df_pl is None:
                continue
            if col not in df_pl.columns:
                raise ValueError(
                    f"Target column '{col}' not found in DataFrames['{split}']; "
                    "cannot apply target transform."
                )
    
            self.DataFrames[split] = df_pl.with_columns(
                self._target_transform_expr(col).alias(col)
            )

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
        Ensures that labels are numeric (int/float/bool).
        """
        import pandas as pd
        from pandas.api.types import is_numeric_dtype

        if TrainData is None:
            raise ValueError("TrainData must not be None for LightGBM.")

        if not NumericColumnNames:
            raise ValueError("NumericColumnNames must be provided for LightGBM.")

        # Helper: ensure label is numeric
        def _ensure_numeric_label(df, target_name):
            if df is None:
                return None
            y = df[target_name]
            if is_numeric_dtype(y):
                return y
            # Try to coerce if it looks numeric
            try:
                y_num = pd.to_numeric(y)
                return y_num
            except Exception:
                raise ValueError(
                    f"LightGBM requires numeric labels; column '{target_name}' has dtype "
                    f"{y.dtype}. Please encode it to integers or floats before training."
                )

        # Features
        def _get_features(df):
            if df is None:
                return None
            return df[NumericColumnNames]

        # Weights
        def _get_weights(df):
            if df is None or not WeightColumnName:
                return None
            return df[WeightColumnName]

        # Construct datasets
        train_X = _get_features(TrainData)
        train_y = _ensure_numeric_label(TrainData, TargetColumnName)
        train_w = _get_weights(TrainData)

        valid_X = _get_features(ValidationData)
        valid_y = _ensure_numeric_label(ValidationData, TargetColumnName) if ValidationData is not None else None
        valid_w = _get_weights(ValidationData)

        test_X = _get_features(TestData)
        test_y = _ensure_numeric_label(TestData, TargetColumnName) if TestData is not None else None
        test_w = _get_weights(TestData)

        def create_dataset(data, label, weight):
            if data is None:
                return None
            return lgbm.Dataset(data=data, label=label, weight=weight)

        train_dataset = create_dataset(train_X, train_y, train_w)
        valid_dataset = create_dataset(valid_X, valid_y, valid_w)
        test_dataset  = create_dataset(test_X,  test_y,  test_w)

        return {
            "train_data": train_dataset,
            "validation_data": valid_dataset,
            "test_data": test_dataset
        }

    # Encode classification / multiclass target to numeric, if not already
    def _encode_target_labels(self):
        """
        Ensure classification / multiclass targets are numeric.

        - If target is already numeric → do nothing.
        - If target is string / object → create mapping {label -> int} using train data,
          apply it to train / validation / test, and store the mapping on self.
        """
        if self.TargetType not in ("classification", "multiclass"):
            return  # regression: nothing to do

        if self.TargetColumnName is None:
            raise RuntimeError("TargetColumnName is None; cannot encode target labels.")

        target = self.TargetColumnName

        train_df = self.DataFrames.get("train")
        if train_df is None:
            raise RuntimeError("Training data is missing; cannot encode target labels.")

        # Get dtype from Polars
        dtype = train_df[target].dtype

        # If already numeric / bool, nothing to do
        numeric_dtypes = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
            pl.Boolean,
        }
        if dtype in numeric_dtypes:
            return

        # Otherwise, build a label mapping from TRAIN ONLY
        unique_labels = train_df[target].unique().to_list()

        # Stable order: sort for reproducibility
        unique_labels_sorted = sorted(unique_labels)

        label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels_sorted)}
        int_to_label = {idx: lbl for lbl, idx in label_to_int.items()}

        self.LabelMapping = label_to_int
        self.LabelMappingInverse = int_to_label

        # Function to apply mapping to a Polars DataFrame
        def _apply_mapping(df: pl.DataFrame | None) -> pl.DataFrame | None:
            if df is None:
                return None
            return df.with_columns(
                pl.col(target).replace(label_to_int).cast(pl.Int64)
            )

        # Apply to all internal splits
        self.DataFrames["train"] = _apply_mapping(self.DataFrames.get("train"))
        self.DataFrames["validation"] = _apply_mapping(self.DataFrames.get("validation"))
        self.DataFrames["test"] = _apply_mapping(self.DataFrames.get("test"))

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
        Threads=-1,
        TargetTransform: str | None = None,
    ):
        """
        Create modeling objects for specific algorithms (CatBoost, XGBoost, LightGBM).
    
        Parameters:
            TrainData, ValidationData, TestData: pandas or polars DataFrames.
            TargetColumnName: Target column name.
            NumericColumnNames, CategoricalColumnNames, TextColumnNames: Lists of column names.
            WeightColumnName: Column name for sample weights.
            Threads: Number of threads to utilize.
            TargetTransform : {"log", "log1p", "sqrt", "standardize", None, "none"}, optional
              Optional target transform for regression:
                - None / "none" → no transform
                - "log"        → log(y)
                - "log1p"      → log(1 + y)
                - "sqrt"       → sqrt(y)
                - "standardize"→ (y - mean) / std  (params learned from TrainData)
      
              Applied in create_model_data() for regression and inverted in score().
    
        Side effects:
            - Stores POLARS originals in self.DataFrames["train"/"validation"/"test"]
            - Stores algo-specific objects in self.ModelData
            - Initializes self.ModelArgs via create_model_parameters()
        """

        # 0) Optionally set / override target transform
        if TargetTransform is not None:
            # this normalizes "none" → None and validates
            self.set_target_transform(TargetTransform)

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

        # 3b) Encode target labels to integers for classification / multiclass
        self._encode_target_labels()

        # Refresh local refs in case target was transformed
        TrainData = self.DataFrames["train"]
        ValidationData = self.DataFrames["validation"]
        TestData = self.DataFrames["test"]

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
    
    # Initialize params
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
        if self.Algorithm == "catboost":
        
            AlgoArgs: dict[str, t.Any] = {}
        
            ###############################
            # TargetType Parameters
            ###############################
            if self.TargetType == "classification":
                AlgoArgs["loss_function"] = "Logloss"
                AlgoArgs["eval_metric"] = "Logloss"
                AlgoArgs["auto_class_weights"] = "Balanced"
        
            elif self.TargetType == "multiclass":
                AlgoArgs["loss_function"] = "MultiClassOneVsAll"
                AlgoArgs["eval_metric"] = "MultiClassOneVsAll"
        
            elif self.TargetType == "regression":
                AlgoArgs["loss_function"] = "RMSE"
                AlgoArgs["eval_metric"] = "RMSE"
        
            else:
                raise ValueError(
                    f"Unsupported TargetType '{self.TargetType}' for CatBoost. "
                    "Choose from 'classification', 'multiclass', 'regression'."
                )
        
            ###############################
            # GPU / CPU Parameters
            ###############################
            if self.GPU:
                AlgoArgs["task_type"] = "GPU"
                AlgoArgs["rsm"] = 1.0
                AlgoArgs["bootstrap_type"] = "Bayesian"
            else:
                AlgoArgs["task_type"] = "CPU"
                AlgoArgs["sampling_frequency"] = "PerTreeLevel"
                AlgoArgs["rsm"] = 0.80
                AlgoArgs["bootstrap_type"] = "MVS"
                AlgoArgs["langevin"] = True
                AlgoArgs["diffusion_temperature"] = 10000
                AlgoArgs["subsample"] = 1.0
        
            ###############################
            # Core Parameters
            ###############################
            AlgoArgs["allow_writing_files"] = False
            AlgoArgs["learning_rate"] = None          # let grid/AutoTune handle
            AlgoArgs["l2_leaf_reg"] = None            # gridable
            AlgoArgs["has_time"] = False
            AlgoArgs["best_model_min_trees"] = 10
            AlgoArgs["nan_mode"] = "Min"
            AlgoArgs["fold_permutation_block"] = 1
            AlgoArgs["boosting_type"] = "Plain"
            AlgoArgs["random_seed"] = None
            AlgoArgs["thread_count"] = -1
            AlgoArgs["metric_period"] = 10
        
            ###############################
            # Gridable Parameters
            ###############################
            AlgoArgs["iterations"] = 1000
            AlgoArgs["depth"] = 6
            AlgoArgs["grow_policy"] = "SymmetricTree"
            AlgoArgs["model_size_reg"] = 0.5
        
            ###############################
            # Dependent Model Parameters
            ###############################
        
            # task_type dependent (kept same for now)
            AlgoArgs["random_strength"] = 1.0
            AlgoArgs["posterior_sampling"] = False
            AlgoArgs["score_function"] = "L2"
            AlgoArgs["border_count"] = 254
        
            # Langevin is typically paired with Bayesian bootstrap & posterior_sampling
            if AlgoArgs.get("langevin", False):
                AlgoArgs["bootstrap_type"] = "Bayesian"
                AlgoArgs["posterior_sampling"] = True
        
            # boost_from_average
            if AlgoArgs["loss_function"] in ["RMSE", "Logloss", "CrossEntropy", "Quantile", "MAE", "MAPE"]:
                AlgoArgs["boost_from_average"] = True
            else:
                AlgoArgs["boost_from_average"] = False

        # XGBoost
        if self.Algorithm == "xgboost":
        
            # Initialize AlgoArgs
            AlgoArgs = dict()
        
            ################################
            # Performance / Environment
            ################################
        
            # Threads: default: use all cores (fall back to -1 if os.cpu_count() fails)
            AlgoArgs["nthread"] = os.cpu_count() or -1
        
            # GPU / CPU tree method & predictor
            if self.GPU:
                # Requires a GPU-enabled XGBoost build
                AlgoArgs["tree_method"] = "gpu_hist"
                AlgoArgs["predictor"] = "gpu_predictor"
            else:
                # CPU defaults
                AlgoArgs["tree_method"] = "hist"
                AlgoArgs["predictor"] = "auto"
        
            # Max number of discrete bins for continuous features
            AlgoArgs["max_bin"] = 256
        
            # Grow policy for trees
            AlgoArgs["grow_policy"] = "depthwise"
        
            ################################
            # Core Booster Parameters
            ################################
            # Learning rate
            AlgoArgs["eta"] = 0.30
        
            # Maximum depth of a tree
            AlgoArgs["max_depth"] = 6
        
            # Minimum sum of instance weight (Hessian) needed in a child
            AlgoArgs["min_child_weight"] = 1
        
            # Maximum delta step we allow each leaf output to be
            AlgoArgs["max_delta_step"] = 0
        
            # Subsample ratio of the training instances
            AlgoArgs["subsample"] = 1.0
        
            # Subsample ratio of columns for each tree
            AlgoArgs["colsample_bytree"] = 1.0
        
            # Subsample ratio of columns for each level
            AlgoArgs["colsample_bylevel"] = 1.0
        
            # Subsample ratio of columns for each split
            AlgoArgs["colsample_bynode"] = 1.0
        
            # L1 regularization term on weights
            AlgoArgs["alpha"] = 0.0
        
            # L2 regularization term on weights
            AlgoArgs["lambda"] = 1.0
        
            # Minimum loss reduction required to make a further partition on a leaf
            AlgoArgs["gamma"] = 0.0
        
            ################################
            # Booster type & parallel trees
            ################################
        
            # Number of parallel trees (1 for standard boosting)
            AlgoArgs["num_parallel_tree"] = 1  # default: 1
        
            # Booster type: leave at default ('gbtree') not usually changed.
            AlgoArgs["booster"] = "gbtree"
        
            ################################
            # Target-dependent parameters
            ################################
            if self.TargetType == "classification":
                AlgoArgs["objective"] = "binary:logistic"
                # You chose AUC as default; totally fine. Could also be "logloss".
                AlgoArgs["eval_metric"] = "auc"
        
            elif self.TargetType == "regression":
                AlgoArgs["objective"] = "reg:squarederror"
                AlgoArgs["eval_metric"] = "rmse"
        
            elif self.TargetType == "multiclass":
                AlgoArgs["objective"] = "multi:softprob"
                AlgoArgs["eval_metric"] = "mlogloss"
                # num_class MUST be set later when you know K
                # e.g. via: update_model_parameters(num_class=K, allow_new=True)
        
            else:
                raise ValueError(
                    f"Unsupported TargetType '{self.TargetType}' for XGBoost. "
                    "Choose from 'classification', 'regression', or 'multiclass'."
                )

        # LightGBM
        if self.Algorithm == "lightgbm":
        
            # Setup Environment
            AlgoArgs = dict()
        
            # Target Dependent Args
            if self.TargetType == "classification":
                AlgoArgs["objective"] = "binary"
                AlgoArgs["metric"] = "auc"
        
            elif self.TargetType == "regression":
                AlgoArgs["objective"] = "regression"
                AlgoArgs["metric"] = "rmse"
        
            elif self.TargetType == "multiclass":
                AlgoArgs["objective"] = "multiclassova"
                AlgoArgs["metric"] = "multi_logloss"
        
            else:
                raise ValueError(
                    f"Unsupported TargetType '{self.TargetType}' for LightGBM. "
                    "Choose from 'classification', 'regression', or 'multiclass'."
                )
        
            # Tuning Args
            AlgoArgs["num_iterations"] = 1000
            AlgoArgs["learning_rate"] = None
            AlgoArgs["num_leaves"] = 31
            AlgoArgs["bagging_freq"] = 0
            AlgoArgs["bagging_fraction"] = 1.0
            AlgoArgs["feature_fraction"] = 1.0
            AlgoArgs["feature_fraction_bynode"] = 1.0
            AlgoArgs["max_delta_step"] = 0.0
        
            # Core Args
            AlgoArgs["task"] = "train"
            AlgoArgs["boosting"] = "gbdt"
            AlgoArgs["lambda_l1"] = 0.0
            AlgoArgs["lambda_l2"] = 0.0
            AlgoArgs["deterministic"] = True
            AlgoArgs["force_col_wise"] = False
            AlgoArgs["force_row_wise"] = False
            AlgoArgs["max_depth"] = None
            AlgoArgs["min_data_in_leaf"] = 20
            AlgoArgs["min_sum_hessian_in_leaf"] = 0.001
            AlgoArgs["extra_trees"] = False
            AlgoArgs["early_stopping_round"] = 10
            AlgoArgs["first_metric_only"] = True
            AlgoArgs["linear_lambda"] = 0.0
            AlgoArgs["min_gain_to_split"] = 0
            AlgoArgs["monotone_constraints"] = None
            AlgoArgs["monotone_constraints_method"] = "advanced"
            AlgoArgs["monotone_penalty"] = 0.0
            AlgoArgs["forcedsplits_filename"] = None
            AlgoArgs["refit_decay_rate"] = 0.90
            AlgoArgs["path_smooth"] = 0.0
        
            # IO / Dataset Parameters
            AlgoArgs["max_bin"] = 255
            AlgoArgs["min_data_in_bin"] = 3
            AlgoArgs["data_random_seed"] = 1
            AlgoArgs["is_enable_sparse"] = True
            AlgoArgs["enable_bundle"] = True
            AlgoArgs["use_missing"] = True
            AlgoArgs["zero_as_missing"] = False
            AlgoArgs["two_round"] = False
        
            # Convert Parameters
            AlgoArgs["convert_model"] = None
            AlgoArgs["convert_model_language"] = "cpp"
        
            # Objective Parameters
            AlgoArgs["boost_from_average"] = True
            AlgoArgs["alpha"] = 0.90
            AlgoArgs["fair_c"] = 1.0
            AlgoArgs["poisson_max_delta_step"] = 0.70
            AlgoArgs["tweedie_variance_power"] = 1.5
            AlgoArgs["lambdarank_truncation_level"] = 30
        
            # Metric Parameters (metric is in Core)
            AlgoArgs["is_provide_training_metric"] = True
            AlgoArgs["eval_at"] = [1, 2, 3, 4, 5]
        
            # Network Parameters
            AlgoArgs["num_machines"] = 1
        
            # GPU / CPU Parameters
            if self.GPU:
                AlgoArgs["device_type"] = "gpu"
                AlgoArgs["num_gpu"] = 1
                # Leave platform/device IDs at defaults (-1 means "auto-detect")
                AlgoArgs["gpu_platform_id"] = -1
                AlgoArgs["gpu_device_id"] = -1
                AlgoArgs["gpu_use_dp"] = True
            else:
                AlgoArgs["device_type"] = "cpu"
                # These GPU params are ignored on CPU but we can omit them for clarity
                AlgoArgs["num_gpu"] = 0

        # Store Model Parameters
        self.ModelArgs = AlgoArgs
        self.ModelArgsNames = [*self.ModelArgs]

    # Update params
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

    # Print params and args
    def print_algo_args(self):
        print(u.print_dict(self.ModelArgs))


    #################################################
    # Function: Train Model
    #################################################

    # Multiclass class counter
    def _infer_num_classes(self):
        """
        Infer number of classes for multiclass classification.
        Prefer LabelMapping if present; otherwise count unique values in train data.
        """
        if self.TargetType not in ("classification", "multiclass"):
            raise RuntimeError("TargetType is not classification/multiclass.")

        # If we already built a mapping, use it
        if self.LabelMapping is not None:
            return len(self.LabelMapping)

        train_df = self.DataFrames.get("train")
        if train_df is None:
            raise RuntimeError("Training data is missing; cannot infer class count.")

        target = self.TargetColumnName
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' not found in training data.")

        return train_df.select(pl.col(target).n_unique()).item()

    # Main training function
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
            
            # If multiclass and classes_count not provided, infer it
            if self.TargetType == "multiclass" and "classes_count" not in self.ModelArgs:
                n_classes = self._infer_num_classes()
                self.ModelArgs["classes_count"] = n_classes

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
    
            # For multiclass, set num_class if not already provided
            if self.TargetType == "multiclass" and "num_class" not in self.ModelArgs:
                n_classes = self._infer_num_classes()
                self.ModelArgs["num_class"] = n_classes

            # Build evaluation list: ONLY train + validation
            evals = []
            if dtrain is not None:
                evals.append((dtrain, "train"))
            if dvalid is not None:
                evals.append((dvalid, "validation"))
    
            num_boost_round = self.ModelArgs.get("num_boost_round", 1000)
            early_stopping_rounds = self.ModelArgs.get("early_stopping_rounds", 50)
            
            # Remove from ModelArgs because xgb.train() does NOT accept these inside params
            params = {k: v for k, v in self.ModelArgs.items()
                      if k not in ("num_boost_round", "early_stopping_rounds")}

            booster = xgb.train(
                params=params,
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

            # For multiclass, set num_class if not already provided
            if self.TargetType == "multiclass" and "num_class" not in self.ModelArgs:
                n_classes = self._infer_num_classes()
                self.ModelArgs["num_class"] = n_classes

            booster = lgbm.train(
                params=self.ModelArgs,
                train_set=train_set,
                valid_sets=valid_sets,
                num_boost_round=num_boost_round
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

    # Helper
    def _normalize_target_transform(self) -> str:
        """
        Return normalized transform name, or 'none' if not set.
        """
        if self.TargetTransform is None:
            return "none"
        return self.TargetTransform.lower()

    # Build inverse transform expression
    def _inverse_target_transform_expr(self, col_name: str) -> pl.Expr:
        """
        Build a Polars expression that inverts the target transform
        on the given prediction column. Used in score().
        """
        t = self._normalize_target_transform()
    
        if t == "none":
            return pl.col(col_name)
    
        if t == "log":
            shift = float(self.TargetTransformParams.get("shift", 0.0))
            # Inverse of log(y + shift) = exp(pred) - shift
            return pl.col(col_name).exp() - shift
    
        if t == "sqrt":
            # Inverse of sqrt(y) = y^2
            return pl.col(col_name) ** 2
    
        if t == "standardize":
            mean = float(self.TargetTransformParams["mean"])
            std = float(self.TargetTransformParams["std"])
            return pl.col(col_name) * std + mean
    
        raise ValueError(f"Unsupported TargetTransform '{t}'.")

    # Catboost helper
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

    # XGBoost helper
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

    # LightGBM helper
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

    # Apply inverse transform
    def _inverse_transform_predictions_inplace(self, df_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Given a scored Polars DataFrame with prediction column Predict_<TargetColumnName>,
        invert the target transform on that prediction column (for regression only).
    
        Returns a new Polars DataFrame (Polars is immutable).
        """
        if self.TargetType.lower() != "regression":
            return df_pl
    
        t = self._normalize_target_transform()
        if t == "none":
            return df_pl
    
        if self.TargetColumnName is None:
            raise RuntimeError(
                "TargetColumnName is None; cannot inverse target transform predictions."
            )
    
        pred_col = f"Predict_{self.TargetColumnName}"
        if pred_col not in df_pl.columns:
            # Nothing to do; silently return df_pl or raise if you prefer
            return df_pl
    
        return df_pl.with_columns(
            self._inverse_target_transform_expr(pred_col).alias(pred_col)
        )

    # Convert classification / multiclass predictions back to original labels
    def decode_predictions(
        self,
        df=None,
        DataName: str | None = None,
        threshold: float = 0.5,
    ):
        """
        Decode encoded target labels and predicted labels back to original values.

        Parameters
        ----------
        df : polars.DataFrame or pandas.DataFrame or None
            Scored data to decode. If None, uses self.ScoredData[DataName].
        DataName : {"train", "validation", "test"} or None
            Name of internally scored dataset in self.ScoredData.
            Ignored if `df` is provided.
        threshold : float
            Threshold for binary classification when converting probabilities
            to predicted class (0 / 1).

        Returns
        -------
        pl.DataFrame
            DataFrame with:
              - original numeric-encoded target still present (unchanged)
              - 'TrueLabel' : original label values (decoded)
              - 'PredictedLabel' : original label values for predictions
                (for classification / multiclass).
        """
        if self.TargetType not in ("classification", "multiclass"):
            raise ValueError(
                "decode_predictions() is only relevant for classification / multiclass."
            )

        if self.LabelMappingInverse is None:
            raise RuntimeError(
                "LabelMappingInverse is None. Target labels were not encoded or "
                "create_model_data() was not called with string labels."
            )

        # Resolve input DataFrame
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to decode_predictions().")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )

        target = self.TargetColumnName
        if target is None:
            raise RuntimeError("TargetColumnName is None; cannot decode labels.")

        if target not in df_pl.columns:
            raise ValueError(
                f"Encoded target column '{target}' not found in data to decode."
            )

        mapping = self.LabelMappingInverse  # dict: int_code -> original_label

        # Decode TRUE label
        df_pl = df_pl.with_columns(
            pl.col(target).replace(mapping).alias("TrueLabel")
        )

        # Decode PREDICTED label
        if self.TargetType == "classification":
            # Expect probability column 'p1' from score()
            if "p1" not in df_pl.columns:
                raise ValueError(
                    "Missing 'p1' column for binary classification predictions. "
                    "Did you run score() first?"
                )

            # Convert p1 -> {0,1} -> original labels
            pred_int_col = (
                (pl.col("p1") >= threshold)
                .cast(pl.Int64)
                .alias("_PredictedClassIdx")
            )

            df_pl = df_pl.with_columns(pred_int_col)
            df_pl = df_pl.with_columns(
                pl.col(target).replace(mapping).alias("TrueLabel")
            )

        elif self.TargetType == "multiclass":
            # Expect columns 'class_0', 'class_1', ..., 'class_{K-1}'
            prob_cols = [c for c in df_pl.columns if c.startswith("class_")]
            if not prob_cols:
                raise ValueError(
                    "No 'class_k' probability columns found for multiclass decoding. "
                    "Did you run score() with a multiclass model?"
                )

            # Sort by index (class_0, class_1, ...)
            def _class_idx(col_name: str) -> int:
                try:
                    return int(col_name.split("_", 1)[1])
                except Exception:
                    return 0

            prob_cols = sorted(prob_cols, key=_class_idx)

            # Argmax over class probabilities → predicted class index
            df_pl = df_pl.with_columns(
                pl.concat_list([pl.col(c) for c in prob_cols])
                .arg_max()
                .alias("_PredictedClassIdx")
            )

            # Map index to original label
            df_pl = df_pl.with_columns(
                pl.col(target).replace(mapping).alias("TrueLabel")
            )

        return df_pl

    # Single instance score
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

    # Main scoring function
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
    
    # Main Evaluation Function
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
        Regression:
            pl.DataFrame with one row per group (or single row if no grouping).

        Binary classification:
            pl.DataFrame with:
                - one row per group with EvalLevel="overall_binary"
                - many rows per group & threshold with EvalLevel="threshold_curve"

        Multiclass classification:
            pl.DataFrame with:
                - one row per group with EvalLevel="overall_multiclass"
                - many rows per (group, class, threshold) with EvalLevel="one_vs_all"
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

        # Build list of groups as DataFrames
        if by_cols:
            group_dfs = df_pl.partition_by(by_cols, maintain_order=True)
        else:
            group_dfs = [df_pl]

        # timestamp + model name
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = FitName or "RetroFitModel"
        group_label = ",".join(by_cols) if by_cols else "GLOBAL"

        # Helper: build key_dict for a group DataFrame
        def _build_key_dict(gdf: pl.DataFrame) -> dict:
            if not by_cols:
                return {}
            first_row = gdf.select(by_cols).row(0)
            return dict(zip(by_cols, first_row))

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

            for gdf in group_dfs:
                if gdf.height == 0:
                    continue

                key_dict = _build_key_dict(gdf)

                y_true = gdf[target].to_numpy()
                y_pred = gdf[pred_col].to_numpy()

                if y_true.size == 0:
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
                    "EvalLevel": "regression",
                    "Threshold": None,
                    "ClassIndex": None,
                    "ClassName": None,
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
            default_thr = 0.5
            all_rows = []

            for gdf in group_dfs:
                if gdf.height == 0:
                    continue

                key_dict = _build_key_dict(gdf)

                y_true = gdf[target].to_numpy()
                p1 = gdf["p1"].to_numpy()

                N1 = len(y_true)
                if N1 == 0:
                    continue

                P1 = np.sum(y_true == 1)
                N0 = N1 - P1

                # -----------------------
                # 3a) Overall metrics at default threshold (0.5)
                # -----------------------
                y_pred_def = (p1 >= default_thr).astype(int)

                TP = int(np.sum((y_pred_def == 1) & (y_true == 1)))
                TN = int(np.sum((y_pred_def == 0) & (y_true == 0)))
                FP = int(np.sum((y_pred_def == 1) & (y_true == 0)))
                FN = int(np.sum((y_pred_def == 0) & (y_true == 1)))

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
                        (P1 / N1) * (tpc * TPR + fpc * (1 - TPR))
                        + (1 - P1 / N1) * (fnc * FPR + tnc * (1 - FPR))
                    )

                overall_row = {
                    "ModelName": model_name,
                    "CreateTime": create_time,
                    "GroupingVars": group_label,
                    "EvalLevel": "overall_binary",
                    "Threshold": default_thr,
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
                    "ClassIndex": None,
                    "ClassName": None,
                }
                overall_row.update(key_dict)
                all_rows.append(overall_row)

                # -----------------------
                # 3b) Threshold curve (as before)
                # -----------------------
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
                            (P1 / N1) * (tpc * TPR + fpc * (1 - TPR))
                            + (1 - P1 / N1) * (fnc * FPR + tnc * (1 - FPR))
                        )

                    row = {
                        "ModelName": model_name,
                        "CreateTime": create_time,
                        "GroupingVars": group_label,
                        "EvalLevel": "threshold_curve",
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
                        "ClassIndex": None,
                        "ClassName": None,
                    }

                    row.update(key_dict)
                    all_rows.append(row)

            out = pl.DataFrame(all_rows) if all_rows else pl.DataFrame([])

            key = FitName or f"{self.Algorithm}_classification_{DataName or 'data'}"
            self.EvaluationList[key] = out
            self.EvaluationListNames.append(key)

            return out

        # -------------------------------
        # 4) Multiclass Classification
        # -------------------------------
        if self.TargetType.lower() == "multiclass":
            # Expect columns 'class_0', 'class_1', ..., 'class_{K-1}' from score()
            prob_cols = [c for c in df_pl.columns if c.startswith("class_")]
            if not prob_cols:
                raise ValueError(
                    "No 'class_k' probability columns found for multiclass evaluation. "
                    "Did you run score() with a multiclass model?"
                )

            def _class_idx(col_name: str) -> int:
                try:
                    return int(col_name.split("_", 1)[1])
                except Exception:
                    return 0

            prob_cols = sorted(prob_cols, key=_class_idx)

            # Cost + thresholds for per-class (one-vs-all) evaluation
            tpc = CostDict.get("tpcost", 0.0)
            fpc = CostDict.get("fpcost", 1.0)
            fnc = CostDict.get("fncost", 1.0)
            tnc = CostDict.get("tncost", 0.0)
            thresholds = np.linspace(0.0, 1.0, 101)

            all_rows = []

            # Optional mapping from class index -> original label
            mapping_inv = getattr(self, "LabelMappingInverse", None)

            for gdf in group_dfs:
                if gdf.height == 0:
                    continue

                key_dict = _build_key_dict(gdf)

                y_true = gdf[target].to_numpy()
                if y_true.size == 0:
                    continue

                probs = gdf[prob_cols].to_numpy()  # shape: (N, K)
                n_classes = probs.shape[1]
                N1 = len(y_true)

                # -----------------------
                # 4a) Overall multiclass metrics
                # -----------------------
                y_pred_overall = probs.argmax(axis=1)

                acc = float(np.mean(y_true == y_pred_overall))
                f1_macro = f1_score(y_true, y_pred_overall, average="macro")
                f1_micro = f1_score(y_true, y_pred_overall, average="micro")
                f1_weighted = f1_score(y_true, y_pred_overall, average="weighted")

                overall_row = {
                    "ModelName": model_name,
                    "CreateTime": create_time,
                    "GroupingVars": group_label,
                    "EvalLevel": "overall_multiclass",
                    "n_obs": int(N1),
                    "Accuracy": acc,
                    "F1_macro": f1_macro,
                    "F1_micro": f1_micro,
                    "F1_weighted": f1_weighted,
                    "Threshold": None,
                    "ClassIndex": None,
                    "ClassName": None,
                }
                overall_row.update(key_dict)
                all_rows.append(overall_row)

                # -----------------------
                # 4b) Per-class one-vs-all evaluation
                # -----------------------
                for class_pos, col_name in enumerate(prob_cols):
                    class_idx = _class_idx(col_name)
                    class_name = (
                        mapping_inv.get(class_idx)
                        if isinstance(mapping_inv, dict)
                        else str(class_idx)
                    )

                    p1 = probs[:, class_pos]  # prob of "this class"
                    y_bin = (y_true == class_idx).astype(int)

                    P1 = int(np.sum(y_bin == 1))
                    N0 = N1 - P1

                    for thr in thresholds:
                        y_pred = (p1 >= thr).astype(int)

                        TP = int(np.sum((y_pred == 1) & (y_bin == 1)))
                        TN = int(np.sum((y_pred == 0) & (y_bin == 0)))
                        FP = int(np.sum((y_pred == 1) & (y_bin == 0)))
                        FN = int(np.sum((y_pred == 0) & (y_bin == 1)))

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
                                (P1 / N1) * (tpc * TPR + fpc * (1 - TPR))
                                + (1 - P1 / N1) * (fnc * FPR + tnc * (1 - FPR))
                            )

                        row = {
                            "ModelName": model_name,
                            "CreateTime": create_time,
                            "GroupingVars": group_label,
                            "EvalLevel": "one_vs_all",
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
                            "ClassIndex": class_idx,
                            "ClassName": class_name,
                        }

                        row.update(key_dict)
                        all_rows.append(row)

            out = pl.DataFrame(all_rows) if all_rows else pl.DataFrame([])

            key = FitName or f"{self.Algorithm}_multiclass_{DataName or 'data'}"
            self.EvaluationList[key] = out
            self.EvaluationListNames.append(key)

            return out

        # -------------------------------
        # 5) Fallback
        # -------------------------------
        raise NotImplementedError(
            f"Evaluation not implemented for TargetType='{self.TargetType}'."
        )


    #################################################
    # Function: Save / Load entire RetroFit object
    #################################################
    
    # Save class object
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

    # Load class object
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


    #################################################
    # Function: Feature Importance
    #################################################
    
    # Main variable importance function
    def compute_feature_importance(
        self,
        ModelName: str | None = None,
        normalize: bool = True,
        sort: bool = True,
        importance_type: str = "gain",
    ) -> pl.DataFrame:
        """
        Compute feature importance for the current algorithm (catboost/xgboost/lightgbm).

        Parameters
        ----------
        ModelName : str or None
            If provided, use that model from self.ModelList / self.FitList.
            If None, use self.Model.
        normalize : bool
            If True, add 'importance_norm' that sums to 1.
        sort : bool
            If True, sort descending by importance and add 'rank'.
        importance_type : str
            - CatBoost: ignored (always uses FeatureImportance).
            - XGBoost: one of {'weight','gain','cover','total_gain','total_cover'}.
            - LightGBM: one of {'split','gain'}.

        Returns
        -------
        pl.DataFrame
            Columns:
                - feature
                - importance
                - importance_norm (if normalize=True)
                - rank (if sort=True)
        """

        # 1) Resolve model
        if self.Model is None and not self.ModelList:
            raise RuntimeError(
                "No trained models found. Call train() before compute_feature_importance()."
            )

        if ModelName is not None:
            model = self.ModelList.get(ModelName) or self.FitList.get(ModelName)
            if model is None:
                raise KeyError(f"Model '{ModelName}' not found in ModelList or FitList.")
        else:
            if self.Model is None:
                raise RuntimeError("self.Model is None and no ModelName provided.")
            model = self.Model

        algo = self.Algorithm  # already lowercased in __init__

        # 2) Determine feature names used by this algorithm
        if algo == "xgboost":
            feature_cols = self.NumericColumnNames or []
        else:
            feature_cols = (self.NumericColumnNames or []) + \
                           (self.CategoricalColumnNames or []) + \
                           (self.TextColumnNames or [])

        if not feature_cols:
            raise RuntimeError(
                "No feature columns found; check Numeric/Categorical/Text column lists."
            )

        # 3) Algorithm-specific importance extraction
        if algo == "catboost":
            importances = np.array(
                model.get_feature_importance(type="FeatureImportance")
            )

            if len(importances) != len(feature_cols):
                raise ValueError(
                    f"CatBoost importance length {len(importances)} != "
                    f"number of feature columns {len(feature_cols)}. "
                    "Check that your feature lists match training order."
                )

            df_imp = pl.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": importances.astype(float),
                }
            )

        elif algo == "xgboost":
            # XGBoost Booster.get_score returns dict: {feature_name_or_fid: score}
            valid_types = {"weight", "gain", "cover", "total_gain", "total_cover"}
            if importance_type not in valid_types:
                raise ValueError(
                    f"importance_type '{importance_type}' not valid for XGBoost. "
                    f"Choose from {valid_types}."
                )

            score_dict = model.get_score(importance_type=importance_type)

            # Helper to detect "f0", "f1", ... style keys
            def _is_f_index(k: str) -> bool:
                return isinstance(k, str) and k.startswith("f") and k[1:].isdigit()

            keys = list(score_dict.keys())
            use_index_keys = bool(keys) and all(_is_f_index(k) for k in keys)

            importances = []

            if use_index_keys:
                # Map feature_cols by position → "f0", "f1", ...
                for idx, feat in enumerate(feature_cols):
                    key = f"f{idx}"
                    val = score_dict.get(key, 0.0)
                    importances.append(val)
            else:
                # Assume keys are actual feature names
                for feat in feature_cols:
                    val = score_dict.get(feat, 0.0)
                    importances.append(val)

            df_imp = pl.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": np.array(importances, dtype=float),
                }
            )

        elif algo == "lightgbm":
            # LightGBM Booster: feature_importance + feature_name
            lgb_type = "gain" if importance_type == "gain" else "split"

            importances = np.array(
                model.feature_importance(importance_type=lgb_type),
                dtype=float
            )
            names = model.feature_name()

            if len(importances) != len(names):
                raise ValueError(
                    f"LightGBM returned {len(importances)} importances for "
                    f"{len(names)} features."
                )

            df_imp = pl.DataFrame(
                {
                    "feature": names,
                    "importance": importances,
                }
            )

        else:
            raise ValueError(
                f"compute_feature_importance not implemented for algorithm '{self.Algorithm}'."
            )

        # 4) Normalize and sort
        if normalize:
            total = df_imp["importance"].sum()
            if total != 0:
                df_imp = df_imp.with_columns(
                    (pl.col("importance") / total).alias("importance_norm")
                )
            else:
                df_imp = df_imp.with_columns(
                    pl.lit(0.0).alias("importance_norm")
                )

        if sort:
            df_imp = df_imp.sort("importance", descending=True)
            df_imp = df_imp.with_columns(
                pl.arange(1, df_imp.height + 1).alias("rank")
            )

        # 5) Store in ImportanceList for downstream plots / reports
        key_name = ModelName or "MainModel"
        imp_key = f"{self.Algorithm}_feature_importance_{key_name}"
        self.ImportanceList[imp_key] = df_imp
        self.ImportanceListNames.append(imp_key)

        return df_imp

    # Main interaction importance function
    def compute_catboost_interaction_importance(
        self,
        ModelName: str | None = None,
        top_n: int | None = None,
        normalize: bool = True,
    ) -> pl.DataFrame:
        """
        Compute pairwise interaction importance for a CatBoost model.

        Uses CatBoost's get_feature_importance(type='Interaction'), which
        returns [feat_idx_1, feat_idx_2, score].

        Parameters
        ----------
        ModelName : str or None
            If provided, use that model from self.ModelList / self.FitList.
            If None, use self.Model.
        top_n : int or None
            If provided, keep only the top_n interactions by importance.
        normalize : bool
            If True, add 'importance_norm' that sums to 1.

        Returns
        -------
        pl.DataFrame with columns:
            - feature_1
            - feature_2
            - importance
            - importance_norm (if normalize=True)
            - rank
        """
        if self.Algorithm != "catboost":
            raise ValueError(
                "compute_catboost_interaction_importance is only available for CatBoost models."
            )

        # Resolve model
        if self.Model is None and not self.ModelList:
            raise RuntimeError(
                "No trained models found. Call train() before compute_catboost_interaction_importance()."
            )

        if ModelName is not None:
            model = self.ModelList.get(ModelName) or self.FitList.get(ModelName)
            if model is None:
                raise KeyError(f"Model '{ModelName}' not found in ModelList or FitList.")
        else:
            if self.Model is None:
                raise RuntimeError("self.Model is None and no ModelName provided.")
            model = self.Model

        # Feature columns used by CatBoost
        feature_cols = (self.NumericColumnNames or []) + \
                       (self.CategoricalColumnNames or []) + \
                       (self.TextColumnNames or [])

        if not feature_cols:
            raise RuntimeError(
                "No feature columns found; check Numeric/Categorical/Text column lists."
            )

        # Get interaction importances
        interactions = np.array(
            model.get_feature_importance(type="Interaction")
        )
        # Expected shape: (n_interactions, 3) → [idx1, idx2, score]
        if interactions.ndim != 2 or interactions.shape[1] < 3:
            raise ValueError(
                "Unexpected shape from CatBoost get_feature_importance(type='Interaction')."
            )

        idx1 = interactions[:, 0].astype(int)
        idx2 = interactions[:, 1].astype(int)
        scores = interactions[:, 2].astype(float)

        max_idx = max(idx1.max(), idx2.max())
        if max_idx >= len(feature_cols):
            raise ValueError(
                "Interaction indices exceed number of feature columns. "
                "Check that feature column lists match training."
            )

        df_int = pl.DataFrame(
            {
                "feature_1": [feature_cols[i] for i in idx1],
                "feature_2": [feature_cols[j] for j in idx2],
                "importance": scores,
            }
        )

        df_int = df_int.sort("importance", descending=True)

        if top_n is not None and top_n > 0 and top_n < df_int.height:
            df_int = df_int.head(top_n)

        if normalize:
            total = df_int["importance"].sum()
            if total != 0:
                df_int = df_int.with_columns(
                    (pl.col("importance") / total).alias("importance_norm")
                )
            else:
                df_int = df_int.with_columns(
                    pl.lit(0.0).alias("importance_norm")
                )

        df_int = df_int.with_columns(
            pl.arange(1, df_int.height + 1).alias("rank")
        )

        # Store in InteractionImportanceList
        key_name = ModelName or "MainModel"
        int_key = f"catboost_interaction_importance_{key_name}"
        self.InteractionImportanceList[int_key] = df_int
        self.InteractionImportanceListNames.append(int_key)

        return df_int
    

    #################################################
    # Function: Calibration Tables
    #################################################

    # Helper
    def _store_calibration_table(self, key: str, table):
        """
        Store a calibration table inside CalibrationList and track the key.
        """
        self.CalibrationList[key] = table
        self.CalibrationListNames.append(key)

    # Regression
    def build_regression_calibration_table(
        self,
        DataName: str | None = None,
        df=None,
        ByVariables=None,
        target: str | None = None,
        pred_col: str | None = None,
        n_bins: int = 20,
        binning: str = "equal_width",   # "equal_width" or "quantile"
        store: bool = True,
    ) -> pl.DataFrame:
        """
        Build calibration data for regression.
    
        You can use either:
          - DataName="test" (uses self.ScoredData["test"] and self.TargetColumnName)
          - df=..., target="y", pred_col="y_hat" for arbitrary scored data
    
        Each row corresponds to a (group, bin) with summary stats.
        """
    
        # ---- Normalize ByVariables ----
        if ByVariables is None:
            by_cols: list[str] = []
        elif isinstance(ByVariables, str):
            by_cols = [ByVariables]
        elif isinstance(ByVariables, (list, tuple)):
            by_cols = list(ByVariables)
        else:
            raise TypeError("ByVariables must be None, a string, or a list/tuple of strings.")
    
        # ---- Resolve scored data (Polars) ----
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to build_regression_calibration_table().")
    
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )
    
        # ---- Resolve target & prediction columns ----
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError(
                    "self.TargetColumnName is None. Supply `target=` when using an external df."
                )
            target = self.TargetColumnName
    
        if pred_col is None:
            pred_col = f"Predict_{target}"
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in scored data.")
        if pred_col not in df_pl.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in scored data.")
    
        if self.TargetType.lower() != "regression":
            raise ValueError("build_regression_calibration_table() is only valid for regression.")
    
        if binning not in ("equal_width", "quantile"):
            raise ValueError("binning must be 'equal_width' or 'quantile'.")
    
        # ---- Assign bins ----
        if binning == "equal_width":
            # Global min/max over predictions
            stats = df_pl.select(
                pl.col(pred_col).min().alias("min_pred"),
                pl.col(pred_col).max().alias("max_pred"),
            ).to_dicts()[0]
    
            min_pred = float(stats["min_pred"])
            max_pred = float(stats["max_pred"])
            if max_pred == min_pred:
                max_pred = min_pred + 1e-9
    
            width = max_pred - min_pred
    
            df_pl = df_pl.with_columns(
                (
                    ((pl.col(pred_col) - min_pred) / width * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            )
    
        else:  # "quantile"
            n_rows = df_pl.height
            if n_rows == 0:
                return pl.DataFrame([])
    
            df_pl = df_pl.with_columns(
                pl.col(pred_col)
                .rank(method="average")
                .alias("__rank__")
            ).with_columns(
                (
                    ((pl.col("__rank__") - 1) / float(n_rows) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            ).drop("__rank__")
    
        # ---- Group & aggregate ----
        group_cols = by_cols + ["bin_id"]
    
        agg_df = (
            df_pl
            .group_by(group_cols, maintain_order=True)
            .agg([
                pl.count().alias("n_obs"),
                pl.col(pred_col).mean().alias("pred_mean"),
                pl.col(target).mean().alias("actual_mean"),
                pl.col(pred_col).min().alias("pred_min_bin"),
                pl.col(pred_col).max().alias("pred_max_bin"),
            ])
        )
    
        # Bin fractions for plotting
        agg_df = agg_df.with_columns([
            (pl.col("bin_id").cast(pl.Float64) / n_bins).alias("bin_frac_lower"),
            ((pl.col("bin_id") + 1).cast(pl.Float64) / n_bins).alias("bin_frac_upper"),
            ((pl.col("bin_id").cast(pl.Float64) + 0.5) / n_bins).alias("bin_center_frac"),
        ])
    
        # Numeric bin bounds in prediction space
        agg_df = agg_df.with_columns([
            pl.col("pred_min_bin").alias("bin_lower"),
            pl.col("pred_max_bin").alias("bin_upper"),
        ])
    
        # ---- Metadata columns ----
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = "RetroFitModel"
        grouping_label = ",".join(by_cols) if by_cols else "GLOBAL"
    
        agg_df = agg_df.with_columns([
            pl.lit(model_name).alias("ModelName"),
            pl.lit(create_time).alias("CreateTime"),
            pl.lit(grouping_label).alias("GroupingVars"),
            pl.lit(f"calibration_regression_{binning}").alias("EvalLevel"),
        ])
    
        front_cols = ["ModelName", "CreateTime", "GroupingVars", "EvalLevel"]
        other_cols = [c for c in agg_df.columns if c not in front_cols]
        agg_df = agg_df.select(front_cols + other_cols)
    
        # 1) Sort by bin_id while numeric
        agg_df = agg_df.sort("bin_id")
    
        # 2) Cast bin boundary fields to string for nicer categorical x-axis
        bin_label_cols = [
            "bin_frac_lower",
            "bin_frac_upper",
            "bin_center_frac",
            "bin_lower",
            "bin_upper",
        ]
        agg_df = agg_df.with_columns([pl.col(c).cast(pl.Utf8) for c in bin_label_cols])
    
        # Optional store
        if store:
            key = f"{self.Algorithm}_regression_calibration_{binning}_{DataName or 'data'}"
            self._store_calibration_table(key, agg_df)
    
        return agg_df

    # Classification Calibration Table
    def build_binary_calibration_table(
        self,
        DataName: str | None = None,
        df=None,
        ByVariables=None,
        target: str | None = None,
        prob_col: str = "p1",
        n_bins: int = 20,
        binning: str = "equal_width",   # "equal_width" or "quantile"
        store: bool = True,
    ) -> pl.DataFrame:
        """
        Build calibration data for binary classification.
    
        You can use either:
          - DataName="test" (uses self.ScoredData["test"], target=self.TargetColumnName, prob_col="p1")
          - df=..., target="y", prob_col="p_hat" for arbitrary scored data
        """
    
        # ---- Normalize ByVariables ----
        if ByVariables is None:
            by_cols: list[str] = []
        elif isinstance(ByVariables, str):
            by_cols = [ByVariables]
        elif isinstance(ByVariables, (list, tuple)):
            by_cols = list(ByVariables)
        else:
            raise TypeError("ByVariables must be None, a string, or a list/tuple of strings.")
    
        # ---- Resolve scored data (Polars) ----
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to build_binary_calibration_table().")
    
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )
    
        # ---- Resolve target & prob columns ----
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError(
                    "self.TargetColumnName is None. Supply `target=` when using an external df."
                )
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in scored data.")
        if prob_col not in df_pl.columns:
            raise ValueError(f"Probability column '{prob_col}' not found in scored data.")
    
        if self.TargetType.lower() != "classification":
            raise ValueError("build_binary_calibration_table() is only valid for binary classification.")
    
        if binning not in ("equal_width", "quantile"):
            raise ValueError("binning must be 'equal_width' or 'quantile'.")
    
        # Clip probabilities into [0, 1]
        df_pl = df_pl.with_columns(
            pl.col(prob_col).clip(0.0, 1.0).alias(prob_col)
        )
    
        # ---- Assign bins ----
        if binning == "equal_width":
            df_pl = df_pl.with_columns(
                (
                    (pl.col(prob_col) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            )
        else:  # "quantile"
            n_rows = df_pl.height
            if n_rows == 0:
                return pl.DataFrame([])
    
            df_pl = df_pl.with_columns(
                pl.col(prob_col)
                .rank(method="average")
                .alias("__rank__")
            ).with_columns(
                (
                    ((pl.col("__rank__") - 1) / float(n_rows) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            ).drop("__rank__")
    
        group_cols = by_cols + ["bin_id"]
    
        agg_df = (
            df_pl
            .group_by(group_cols, maintain_order=True)
            .agg([
                pl.count().alias("n_obs"),
                pl.col(prob_col).mean().alias("pred_mean"),
                pl.col(target).mean().alias("actual_mean"),
                pl.col(prob_col).min().alias("pred_min_bin"),
                pl.col(prob_col).max().alias("pred_max_bin"),
            ])
        )
    
        # Fractions & bounds
        agg_df = agg_df.with_columns([
            (pl.col("bin_id").cast(pl.Float64) / n_bins).alias("bin_frac_lower"),
            ((pl.col("bin_id") + 1).cast(pl.Float64) / n_bins).alias("bin_frac_upper"),
            ((pl.col("bin_id").cast(pl.Float64) + 0.5) / n_bins).alias("bin_center_frac"),
            pl.col("pred_min_bin").alias("bin_lower"),
            pl.col("pred_max_bin").alias("bin_upper"),
        ])
    
        # ---- Metadata ----
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = "RetroFitModel"
        grouping_label = ",".join(by_cols) if by_cols else "GLOBAL"
    
        agg_df = agg_df.with_columns([
            pl.lit(model_name).alias("ModelName"),
            pl.lit(create_time).alias("CreateTime"),
            pl.lit(grouping_label).alias("GroupingVars"),
            pl.lit(f"calibration_binary_{binning}").alias("EvalLevel"),
        ])
    
        front_cols = ["ModelName", "CreateTime", "GroupingVars", "EvalLevel"]
        other_cols = [c for c in agg_df.columns if c not in front_cols]
        agg_df = agg_df.select(front_cols + other_cols)
    
        # 1) Sort by bin_id
        agg_df = agg_df.sort("bin_id")
    
        # 2) Cast bin boundary fields to string (for nice categorical axes)
        bin_label_cols = [
            "bin_frac_lower",
            "bin_frac_upper",
            "bin_center_frac",
            "bin_lower",
            "bin_upper",
        ]
        agg_df = agg_df.with_columns([pl.col(c).cast(pl.Utf8) for c in bin_label_cols])
    
        # Optional store
        if store:
            key = f"{self.Algorithm}_binary_calibration_{binning}_{DataName or 'data'}"
            self._store_calibration_table(key, agg_df)
    
        return agg_df


    #################################################
    # Function: Evaluation Plots
    #################################################

    # Helper for regression calibration plot
    @staticmethod
    def _get_regression_calibration_metrics(
        df_pl: pl.DataFrame,
        target: str,
        pred_col: str,
        n_bins: int = 20,
        binning: str = "quantile",
    ) -> dict:
        """
        STATIC HELPER:
          Computes calibration summary metrics (RMSE, MACE) for regression
          given a scored dataframe and column names.
    
        Parameters
        ----------
        df_pl : pl.DataFrame
            Scored dataframe containing target and prediction columns.
        target : str
            Name of the actual/target column.
        pred_col : str
            Name of the prediction column.
        n_bins : int, default 20
            Number of bins for calibration.
        binning : {"equal_width", "quantile"}, default "quantile"
            Strategy for binning predictions.
    
        Returns
        -------
        dict with:
            - "rmse": float
            - "mace": float
        """
        if df_pl is None or df_pl.height == 0:
            raise ValueError("df_pl is empty or None in _get_regression_calibration_metrics().")
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
    
        if pred_col not in df_pl.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in dataframe.")
    
        if binning not in ("equal_width", "quantile"):
            raise ValueError("binning must be 'equal_width' or 'quantile'.")
    
        # -------------------------
        # 1) RMSE (point-wise)
        # -------------------------
        rmse = float(
            df_pl.select(
                ((pl.col(target) - pl.col(pred_col)) ** 2).mean().alias("rmse")
            )["rmse"][0] ** 0.5
        )
    
        # -------------------------
        # 2) MACE via internal calibration binning
        # -------------------------
        n_rows = df_pl.height
        if n_rows == 0:
            return {"rmse": rmse, "mace": float("nan")}
    
        if binning == "equal_width":
            # Global min/max over predictions
            stats = df_pl.select(
                pl.col(pred_col).min().alias("min_pred"),
                pl.col(pred_col).max().alias("max_pred"),
            ).to_dicts()[0]
    
            min_pred = float(stats["min_pred"])
            max_pred = float(stats["max_pred"])
            if max_pred == min_pred:
                max_pred = min_pred + 1e-9
    
            width = max_pred - min_pred
    
            df_bins = df_pl.with_columns(
                (
                    ((pl.col(pred_col) - min_pred) / width * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            )
        else:  # "quantile"
            df_bins = df_pl.with_columns(
                pl.col(pred_col)
                .rank(method="average")
                .alias("__rank__")
            ).with_columns(
                (
                    ((pl.col("__rank__") - 1) / float(n_rows) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            ).drop("__rank__")
    
        cal = (
            df_bins
            .group_by(["bin_id"], maintain_order=True)
            .agg([
                pl.col(pred_col).mean().alias("pred_mean"),
                pl.col(target).mean().alias("actual_mean"),
            ])
        )
    
        mace = float(
            cal.select(
                (pl.col("actual_mean") - pl.col("pred_mean"))
                .abs()
                .mean()
                .alias("mace")
            )["mace"][0]
        )
    
        return {"rmse": rmse, "mace": mace}

    # Helper for regression metrics
    @staticmethod
    def _compute_regression_core_metrics(
        df_pl: pl.DataFrame,
        target: str,
        pred_col: str,
    ) -> pl.DataFrame:
        """
        STATIC HELPER:
          Core regression metrics used across evaluation & plots.
    
        Returns a single-row Polars DataFrame with:
          - RMSE
          - MAE
          - R2
          - MSE
          - VarY  (variance of target)
          - n_obs
        """
        if df_pl is None or df_pl.height == 0:
            raise ValueError("Dataframe is empty or None in _compute_regression_core_metrics().")
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
    
        if pred_col not in df_pl.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in dataframe.")
    
        metrics = df_pl.select(
            [
                # MSE between target and prediction
                ((pl.col(target) - pl.col(pred_col)) ** 2).mean().alias("MSE"),
    
                # MAE
                (pl.col(target) - pl.col(pred_col)).abs().mean().alias("MAE"),
    
                # Variance of target (SST / n)
                ((pl.col(target) - pl.col(target).mean()) ** 2).mean().alias("VarY"),
    
                # Number of observations
                pl.len().alias("n_obs"),
            ]
        )
    
        mse = float(metrics["MSE"][0])
        mae = float(metrics["MAE"][0])
        var_y = float(metrics["VarY"][0])
        n_obs = int(metrics["n_obs"][0])
    
        if var_y > 0:
            r2 = 1.0 - mse / var_y
        else:
            # Degenerate case: constant target → no variance
            r2 = 0.0
    
        rmse = mse ** 0.5
    
        return pl.DataFrame(
            {
                "RMSE":  [rmse],
                "MAE":   [mae],
                "R2":    [r2],
                "MSE":   [mse],
                "VarY":  [var_y],
                "n_obs": [n_obs],
            }
        )

    # Helper for classification calibration plot
    @staticmethod
    def _get_classification_calibration_metrics(
        df_pl: pl.DataFrame,
        target: str,
        prob_col: str = "p1",
        n_bins: int = 20,
        binning: str = "quantile",
    ) -> dict:
        """
        STATIC HELPER:
          Compute summary calibration metrics for a binary classifier:
          - Brier score
          - MACE (mean absolute calibration error)
    
        Parameters
        ----------
        df_pl : pl.DataFrame
            Scored dataframe with target and probability column.
        target : str
            Name of the binary target column (0/1).
        prob_col : str, default "p1"
            Name of the predicted probability column.
        n_bins : int, default 20
            Number of calibration bins.
        binning : {"equal_width", "quantile"}, default "quantile"
            Strategy for binning probabilities.
    
        Returns
        -------
        dict with:
            - "brier": float
            - "mace" : float
        """
        if df_pl is None or df_pl.height == 0:
            raise ValueError("df_pl is empty or None in _get_classification_calibration_metrics().")
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        if prob_col not in df_pl.columns:
            raise ValueError(f"Probability column '{prob_col}' not found in data.")
    
        if binning not in ("equal_width", "quantile"):
            raise ValueError("binning must be 'equal_width' or 'quantile'.")
    
        # Clip probabilities into [0, 1]
        df_pl = df_pl.with_columns(
            pl.col(prob_col).clip(0.0, 1.0).alias(prob_col)
        )
    
        # -------------------------
        # 1) Brier score
        # -------------------------
        brier = float(
            df_pl.select(
                ((pl.col(prob_col) - pl.col(target)) ** 2)
                .mean()
                .alias("brier")
            )["brier"][0]
        )
    
        # -------------------------
        # 2) MACE via internal calibration binning
        # -------------------------
        n_rows = df_pl.height
        if n_rows == 0:
            return {"brier": brier, "mace": float("nan")}
    
        if binning == "equal_width":
            df_bins = df_pl.with_columns(
                (
                    (pl.col(prob_col) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            )
        else:  # "quantile"
            df_bins = df_pl.with_columns(
                pl.col(prob_col)
                .rank(method="average")
                .alias("__rank__")
            ).with_columns(
                (
                    ((pl.col("__rank__") - 1) / float(n_rows) * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("bin_id")
            ).drop("__rank__")
    
        cal = (
            df_bins
            .group_by(["bin_id"], maintain_order=True)
            .agg([
                pl.col(prob_col).mean().alias("pred_mean"),
                pl.col(target).mean().alias("actual_mean"),
            ])
        )
    
        mace = float(
            cal.select(
                (pl.col("actual_mean") - pl.col("pred_mean"))
                .abs()
                .mean()
                .alias("mace")
            )["mace"][0]
        )
    
        return {"brier": brier, "mace": mace}

    # Classification ROC Table
    def _build_binary_roc_table(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
    ) -> tuple[pl.DataFrame, float]:
        """
        PRIVATE:
          Build ROC curve table and compute AUC for binary classification.
    
        Returns
        -------
        roc_df : pl.DataFrame with columns:
            - fpr        (float)
            - tpr        (float)
            - threshold  (float or NaN)
            - fpr_label  (str, rounded for plotting)
        auc_val : float
        """
        if self.TargetType.lower() != "classification":
            raise ValueError(
                "_build_binary_roc_table is only valid when TargetType='classification'."
            )
    
        # 1) Resolve data
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df.")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None; run score(DataName='{DataName}') first."
                )
    
        # 2) Resolve target
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError(
                    "self.TargetColumnName is None; supply target= for external df."
                )
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        if prob_col not in df_pl.columns:
            raise ValueError(f"Probability column '{prob_col}' not found in data.")
    
        # 3) Extract arrays
        y_true = (
            df_pl
            .select(pl.col(target).cast(pl.Int64))
            .to_numpy()
            .ravel()
        )
        y_score = (
            df_pl
            .select(pl.col(prob_col).cast(pl.Float64))
            .to_numpy()
            .ravel()
        )
    
        y_score = np.clip(y_score, 1e-15, 1 - 1e-15)
    
        if y_true.size == 0:
            raise ValueError("No rows available to compute ROC curve.")
    
        # 4) ROC curve + AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_val = float(auc(fpr, tpr))
    
        # 5) Build Polars DataFrame with numeric columns
        roc_df = pl.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            }
        ).sort("fpr")
        
        # 6) Add pretty string labels for x-axis using native Polars
        roc_df = roc_df.with_columns(
            pl.col("fpr")
            .round(4)            # numeric rounding
            .cast(pl.Utf8)       # turn into string for categorical x-axis
            .alias("fpr_label")
        )
    
        return roc_df, auc_val

    # Classification ROC Table
    def _build_binary_pr_table(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
    ) -> tuple[pl.DataFrame, float]:
        """
        PRIVATE:
          Build Precision-Recall curve table & compute PR AUC (AUPRC).
        """
        if self.TargetType.lower() != "classification":
            raise ValueError(
                "_build_binary_pr_table is only valid when TargetType='classification'."
            )
    
        # Resolve data
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df.")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None; run score() first."
                )
    
        # Resolve target
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError("self.TargetColumnName is None; supply target= for external df.")
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found.")
        if prob_col not in df_pl.columns:
            raise ValueError(f"Probability column '{prob_col}' not found.")
    
        y_true = df_pl.select(pl.col(target).cast(pl.Int64)).to_numpy().ravel()
        y_score = df_pl.select(pl.col(prob_col).cast(pl.Float64)).to_numpy().ravel()
    
        y_score = np.clip(y_score, 1e-15, 1 - 1e-15)
    
        # PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall, precision))
    
        pr_df = pl.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": list(thresholds) + [None],  # PR curve trick; len+1
            }
        ).sort("recall")
    
        # Add string labels for x-axis
        pr_df = pr_df.with_columns(
            pl.col("recall")
            .round(4)
            .cast(pl.Utf8)
            .alias("recall_label")
        )
    
        return pr_df, pr_auc

    # Confusion Matrix
    def _build_confusion_matrix(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
        threshold: float = 0.5,
    ):
        """
        PRIVATE:
          Build a confusion matrix for classification / multiclass.
    
        For TargetType='classification' (binary):
          - uses prob_col (default 'p1') and threshold to derive predictions.
    
        For TargetType='multiclass':
          - expects 'class_0', 'class_1', ... probability columns created by score()
            and uses argmax to derive predictions.
    
        Works with either:
          - internal scored data: self.ScoredData[DataName], or
          - external df passed via df=..., with explicit target (and prob_col for binary).
    
        Returns
        -------
        pl.DataFrame in long format with columns:
          - actual         (int)
          - predicted      (int)
          - count          (int)
          - actual_name    (str, if LabelMappingInverse available, else str(actual))
          - predicted_name (str, if LabelMappingInverse available, else str(predicted))
        """
        if self.TargetType not in ("classification", "multiclass"):
            raise ValueError(
                "_build_confusion_matrix is only implemented for TargetType "
                "in {'classification','multiclass'}."
            )
    
        # --- Resolve data ---
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df.")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None; run score(DataName='{DataName}') first, "
                    "or pass an external df."
                )
    
        # --- Resolve target column ---
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError(
                    "self.TargetColumnName is None; supply target= for external df."
                )
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
    
        # --- Resolve predictions depending on TargetType ---
        # y_true is always integer-coded labels (you already encode labels in create_model_data)
        y_true = (
            df_pl
            .select(pl.col(target).cast(pl.Int64))
            .to_numpy()
            .ravel()
        )
    
        if y_true.size == 0:
            raise ValueError("No rows available to build confusion matrix.")
    
        if self.TargetType == "classification":
            # Binary: threshold on prob_col (usually 'p1')
            if prob_col not in df_pl.columns:
                raise ValueError(f"Probability column '{prob_col}' not found in data.")
    
            y_score = (
                df_pl
                .select(pl.col(prob_col).cast(pl.Float64))
                .to_numpy()
                .ravel()
            )
            y_pred = (y_score >= threshold).astype(int)
    
        else:  # multiclass
            prob_cols = [c for c in df_pl.columns if c.startswith("class_")]
            if not prob_cols:
                raise ValueError(
                    "No 'class_k' probability columns found for multiclass confusion matrix. "
                    "Did you run score() with a multiclass model?"
                )
    
            # Keep same order as in evaluate()
            def _class_idx(col_name: str) -> int:
                try:
                    return int(col_name.split("_", 1)[1])
                except Exception:
                    return 0
    
            prob_cols = sorted(prob_cols, key=_class_idx)
    
            probs = df_pl[prob_cols].to_numpy()  # shape (N, K)
            y_pred = probs.argmax(axis=1)
    
        # --- Build long-format confusion table ---
        cm_df = pl.DataFrame(
            {
                "actual": y_true,
                "predicted": y_pred,
            }
        ).group_by(["actual", "predicted"], maintain_order=True).agg(
            pl.len().alias("count")
        )
    
        # --- Attach human-readable labels if available ---
        mapping_inv = getattr(self, "LabelMappingInverse", None)
    
        if isinstance(mapping_inv, dict) and mapping_inv:
            cm_df = cm_df.with_columns(
                pl.col("actual").apply(
                    lambda x: str(mapping_inv.get(int(x), x))
                ).alias("actual_name"),
                pl.col("predicted").apply(
                    lambda x: str(mapping_inv.get(int(x), x))
                ).alias("predicted_name"),
            )
        else:
            cm_df = cm_df.with_columns(
                pl.col("actual").cast(pl.Utf8).alias("actual_name"),
                pl.col("predicted").cast(pl.Utf8).alias("predicted_name"),
            )
    
        return cm_df

    # Regression Calibration Plot
    def plot_regression_calibration(
        self,
        DataName: str | None = None,
        df=None,
        pred_col: str | None = None,
        target: str | None = None,
        n_bins: int = 20,
        binning: str = "quantile",
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Build a regression calibration table (via build_regression_calibration_table)
        and render a QuickEcharts Line plot.
    
        Can work off internal scored data (DataName) or an external df.
    
        Returns
        -------
        dict with:
          - "table"  : Polars DataFrame (calibration table)
          - "metrics": Polars DataFrame with RMSE, MAE, R2, MSE, VarY, n_obs
          - "plot"   : QuickEcharts Line object
        """
    
        # -------------------------
        # Resolve scored data
        # -------------------------
        if df is not None:
            df_pl = self._normalize_input_df(df)
            data_label = "data"
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to plot_regression_calibration().")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None: run score() first."
                )
            data_label = DataName
    
        # Target & prediction column defaults
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError("self.TargetColumnName is None: did you call create_model_data()?")
            target = self.TargetColumnName
    
        if pred_col is None:
            pred_col = f"Predict_{target}"
    
        # -------------------------
        # 1) Build calibration table
        # -------------------------
        cal = self.build_regression_calibration_table(
            DataName=data_label if df is None else None,
            df=df_pl,
            target=target,
            pred_col=pred_col,
            n_bins=n_bins,
            binning=binning,
            store=False,  # plotting call doesn’t need to store by default
        )
    
        # -------------------------
        # 2) Metrics for subtitle
        #    - Core metrics: RMSE, MAE, R² (DataFrame)
        #    - Calibration metrics: RMSE, MACE (dict)
        # -------------------------
        core_metrics = self._compute_regression_core_metrics(
            df_pl=df_pl,
            target=target,
            pred_col=pred_col,
        )
    
        rmse = float(core_metrics["RMSE"][0])
        mae  = float(core_metrics["MAE"][0])
        r2   = float(core_metrics["R2"][0])
    
        cal_metrics = self._get_regression_calibration_metrics(
            df_pl=df_pl,
            target=target,
            pred_col=pred_col,
            n_bins=n_bins,
            binning=binning,
        )
        mace = float(cal_metrics["mace"])
    
        subtitle = (
            f"RMSE = {rmse:.4f} · MAE = {mae:.4f} · R² = {r2:.4f} · "
            f"MACE = {mace:.4f}"
        )
    
        # -------------------------
        # 3) Plot via QuickEcharts
        # -------------------------
        chart = Charts.Line(
            dt=cal,
            PreAgg=True,
            YVar=["pred_mean", "actual_mean"],
            XVar="bin_center_frac",
            RenderHTML=plot_name,
            Theme=Theme,
            Title=f"Regression Calibration ({data_label or 'data'}, {binning})",
            SubTitle=subtitle,
            YAxisTitle="Predicted & Actual",
            XAxisTitle=(
                "Quantile Bins of Predicted Values"
                if binning == "quantile"
                else "Equal-Width Bins of Predicted Values"
            ),
        )
    
        return {"table": cal, "metrics": core_metrics, "plot": chart}

    # Classification Calibration Plot
    def plot_classification_calibration(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
        n_bins: int = 20,
        binning: str = "quantile",
        Store: bool = False,
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Build and visualize a binary classification calibration plot using QuickEcharts.
    
        Can work off:
          - internal scored data (DataName)
          - external df + target + prob_col
    
        Returns
        -------
        dict with:
          - "table"  : Polars DataFrame (calibration table)
          - "metrics": dict with keys {"brier", "mace"}
          - "plot"   : QuickEcharts Line object
        """
        if self.TargetType.lower() != "classification":
            raise ValueError(
                "plot_classification_calibration is only valid when TargetType='classification'."
            )
    
        # 1) Resolve data
        if df is not None:
            df_pl = self._normalize_input_df(df)
            data_key = "data"
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df.")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None; run score(DataName='{DataName}') first."
                )
            data_key = DataName
    
        # 2) Resolve target
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError("self.TargetColumnName is None; supply target= for external df.")
            target = self.TargetColumnName
    
        # 3) Build calibration table
        cal = self.build_binary_calibration_table(
            DataName=data_key if df is None else None,
            df=df_pl,
            target=target,
            prob_col=prob_col,
            ByVariables=None,
            n_bins=n_bins,
            binning=binning,
            store=Store,
        )
    
        # 4) Calibration metrics (Brier, MACE) via static helper
        metrics = self._get_classification_calibration_metrics(
            df_pl=df_pl,
            target=target,
            prob_col=prob_col,
            n_bins=n_bins,
            binning=binning,
        )
    
        subtitle = f"Brier = {metrics['brier']:.4f} · MACE = {metrics['mace']:.4f}"
    
        # 5) Ensure sorted by bin_id for plotting
        cal = cal.sort("bin_id")
    
        # 6) Plot with QuickEcharts
        chart = Charts.Line(
            dt=cal,
            PreAgg=True,
            YVar=["pred_mean", "actual_mean"],
            XVar="bin_center_frac",
            Title=f"Classification Calibration ({data_key})",
            SubTitle=subtitle,
            YAxisTitle="Observed Positive Rate & Mean Predicted Probability",
            XAxisTitle=(
                "Quantile Bins of Predicted Probabilities"
                if binning == "quantile"
                else "Equal-Width Bins of Predicted Probabilities"
            ),
            Theme=Theme,
            RenderHTML=plot_name,
        )
    
        return {"table": cal, "metrics": metrics, "plot": chart}

    # Regression Predicted vs Actual Scatterplot
    def plot_regression_scatter(
        self,
        DataName: str = "test",
        df=None,
        target: str | None = None,
        pred_col: str | None = None,
        SampleSize: int = 15000,
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Actual vs Predicted scatter plot for regression models.
    
        Can use:
          - internal scored data: self.ScoredData[DataName], or
          - external df passed via df=..., with explicit target/pred_col.
    
        Parameters
        ----------
        DataName : str, default "test"
            Key for self.ScoredData[...] when df is not supplied.
        df : pl.DataFrame | pd.DataFrame | None
            Optional external scored data. If provided, this takes precedence
            over self.ScoredData[DataName].
        target : str | None
            Name of the actual/target column. If None, falls back to
            self.TargetColumnName when available.
        pred_col : str | None
            Name of the prediction column. If None, defaults to
            f"Predict_{target}".
        SampleSize : int, default 15000
            Subsample size for the scatter plot.
        plot_name : str | None
            Optional HTML file name for QuickEcharts rendering.
        Theme : str, default "dark"
            Theme for QuickEcharts.
    
        Returns
        -------
        dict with:
          - "table": Polars DataFrame with RMSE, MAE, R2, MSE, VarY, n_obs
          - "plot" : QuickEcharts Scatter object
        """
        if self.TargetType.lower() != "regression":
            raise ValueError(
                "plot_regression_scatter is only valid when TargetType='regression'."
            )
    
        # ------------------------------------------------------------------
        # 1) Resolve data source (external df vs internal scored data)
        # ------------------------------------------------------------------
        if df is not None:
            # allow pandas or polars, assuming the class has a helper for this
            df_pl = self._ensure_polars(df) if hasattr(self, "_ensure_polars") else df
            source_label = "external"
        else:
            df_pl = self.ScoredData.get(DataName)
            source_label = DataName
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None; run score(DataName='{DataName}') first, "
                    "or pass an external df."
                )
    
        # ------------------------------------------------------------------
        # 2) Resolve target and prediction column names
        # ------------------------------------------------------------------
        if target is None:
            target = self.TargetColumnName
        if target is None:
            raise RuntimeError(
                "Target column is None. Provide target=... explicitly or ensure "
                "self.TargetColumnName is set (e.g., via create_model_data())."
            )
    
        if pred_col is None:
            pred_col = f"Predict_{target}"
    
        # safety checks
        cols = set(df_pl.columns)
        missing = [c for c in (target, pred_col) if c not in cols]
        if missing:
            raise ValueError(
                f"The following required columns are missing from the dataframe: {missing}. "
                f"Available columns: {sorted(cols)}"
            )
    
        # ------------------------------------------------------------------
        # 3) Core metrics via shared helper
        # ------------------------------------------------------------------
        metrics = self._compute_regression_core_metrics(df_pl, target, pred_col)
        rmse = float(metrics["RMSE"][0])
        mae  = float(metrics["MAE"][0])
        r2   = float(metrics["R2"][0])
    
        subtitle = f"RMSE = {rmse:,.4f} · MAE = {mae:,.4f} · R² = {r2:,.4f}"
    
        # ------------------------------------------------------------------
        # 4) Build scatter plot
        # ------------------------------------------------------------------
        chart = Charts.Scatter(
            dt=df_pl,
            SampleSize=SampleSize,
            YVar=target,
            XVar=pred_col,
            RenderHTML=plot_name,
            Theme=Theme,
            Title=f"Actual vs Predicted ({source_label})",
            SubTitle=subtitle,
            YAxisTitle="Actual",
            XAxisTitle="Predicted",
        )
    
        return {
            "table": metrics,
            "plot": chart,
        }

    # ROC Curve
    def plot_classification_roc(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
        plot_name: str | None = None,
        Theme: str = "dark",
        GradientColors: list[str] | None = None,
    ):
        """
        Plot ROC curve for binary classification using QuickEcharts.
    
        Can work off:
          - internal scored data (DataName), or
          - external df with target + prob_col.
    
        Returns
        -------
        dict with:
          - "table": pl.DataFrame with columns ["fpr", "tpr", "threshold"]
          - "auc": float, AUC of ROC curve
          - "plot": QuickEcharts Line chart object
        """
        if self.TargetType.lower() != "classification":
            raise ValueError(
                "plot_classification_roc is only valid when TargetType='classification'."
            )

        # Assign values if None
        if GradientColors is None:
            GradientColors = ["#e12191", "#0011FF"]

        # Build ROC table + AUC
        roc_df, auc_val = self._build_binary_roc_table(
            DataName=DataName if df is None else None,
            df=df,
            target=target,
            prob_col=prob_col,
        )
    
        # Title pieces
        data_label = DataName or "data"
        title = f"ROC Curve ({data_label})"
        subtitle = f"AUC = {auc_val:.4f}"
    
        # QuickEcharts Area plot: TPR vs FPR
        chart = Charts.Area(
            dt=roc_df,
            PreAgg=True,
            YVar=["tpr"],
            XVar="fpr_label",
            GradientColors=GradientColors,
            RenderHTML=plot_name,
            Theme=Theme,
            Title=title,
            SubTitle=subtitle,
            YAxisTitle="True Positive Rate (TPR)",
            XAxisTitle="False Positive Rate (FPR)",
        )
    
        return {
            "table": roc_df,
            "auc": auc_val,
            "plot": chart,
        }

    # PR Curve
    def plot_classification_pr(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
        plot_name: str | None = None,
        Theme: str = "dark",
        GradientColors: list[str] | None = None,
    ):
        """
        Plot Precision-Recall curve using QuickEcharts.
        """
        if GradientColors is None:
            GradientColors = ["#e12191", "#0011FF"]
    
        pr_df, pr_auc = self._build_binary_pr_table(
            DataName=DataName if df is None else None,
            df=df,
            target=target,
            prob_col=prob_col,
        )
    
        title = f"Precision-Recall Curve ({DataName or 'data'})"
        subtitle = f"AUPRC = {pr_auc:.4f}"
    
        chart = Charts.Area(
            dt=pr_df,
            PreAgg=True,
            YVar=["precision"],
            XVar="recall_label",      # <<--- string label version
            GradientColors=GradientColors,
            RenderHTML=plot_name,
            Theme=Theme,
            Title=title,
            SubTitle=subtitle,
            YAxisTitle="Precision",
            XAxisTitle="Recall",
        )
    
        return {
            "table": pr_df,
            "auc": pr_auc,
            "plot": chart,
        }


    # Heatmap for Confusion Matrix
    def plot_confusion_matrix(
        self,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        prob_col: str = "p1",
        threshold: float = 0.5,
        normalize: str = "none",  # "none", "true", "pred", "all"
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Plot confusion matrix as a heatmap using QuickEcharts.
    
        Supports:
          - TargetType='classification' (binary, using prob_col + threshold)
          - TargetType='multiclass'     (using argmax over 'class_k' columns)
    
        Parameters
        ----------
        DataName : str | None
            Internal scored dataset key ('train','validation','test').
            Ignored if df is provided.
        df : pl.DataFrame | pd.DataFrame | None
            External scored data with target + score columns.
        target : str | None
            Target column name; defaults to self.TargetColumnName.
        prob_col : str
            Probability column for the positive class (binary case only).
        threshold : float
            Probability threshold for positive class in binary case.
        normalize : {"none","true","pred","all"}
            - "none": use raw counts
            - "true": normalize per actual class (row)
            - "pred": normalize per predicted class (column)
            - "all" : normalize by total count
        plot_name : str | None
            Optional HTML file name for QuickEcharts rendering.
        Theme : str
            QuickEcharts theme.
    
        Returns
        -------
        dict with:
          - "table": pl.DataFrame with [actual, predicted, count, value, actual_name, predicted_name]
          - "plot" : QuickEcharts Heatmap object
        """
        if self.TargetType not in ("classification", "multiclass"):
            raise ValueError(
                "plot_confusion_matrix is only implemented for TargetType "
                "in {'classification','multiclass'}."
            )
    
        # 1) Build confusion matrix table
        cm_df = self._build_confusion_matrix(
            DataName=DataName if df is None else None,
            df=df,
            target=target,
            prob_col=prob_col,
            threshold=threshold,
        )
    
        # 2) Normalize if requested
        if normalize not in ("none", "true", "pred", "all"):
            raise ValueError("normalize must be one of: 'none', 'true', 'pred', 'all'.")
    
        cm_out = cm_df
    
        if normalize == "none":
            cm_out = cm_out.with_columns(
                pl.col("count").cast(pl.Float64).alias("value")
            )
    
        elif normalize == "true":
            # normalize per actual
            cm_out = (
                cm_out
                .with_columns(
                    pl.sum("count").over("actual").alias("_row_total")
                )
                .with_columns(
                    (pl.col("count") / pl.col("_row_total")).alias("value")
                )
                .drop("_row_total")
            )
    
        elif normalize == "pred":
            # normalize per predicted
            cm_out = (
                cm_out
                .with_columns(
                    pl.sum("count").over("predicted").alias("_col_total")
                )
                .with_columns(
                    (pl.col("count") / pl.col("_col_total")).alias("value")
                )
                .drop("_col_total")
            )
    
        else:  # "all"
            total = cm_out["count"].sum()
            total = float(total) if total is not None else 0.0
            if total <= 0:
                total = 1.0
            cm_out = cm_out.with_columns(
                (pl.col("count") / total).alias("value")
            )
    
        # 3) Titles
        data_label = DataName or "data"
        if self.TargetType == "classification":
            tt_label = "Binary Classification"
        else:
            tt_label = "Multiclass Classification"
    
        norm_label = {
            "none": "Counts",
            "true": "Row-normalized (per actual)",
            "pred": "Column-normalized (per predicted)",
            "all": "Global-normalized",
        }[normalize]
    
        title = f"Confusion Matrix ({tt_label}, {data_label})"
        subtitle = norm_label
    
        # 4) Heatmap plot (actual on Y, predicted on X)
        # QuickEcharts heatmap is assumed to use long-form dt with:
        #   - XVar (predicted_name), YVar (actual_name), ValueVar="value"
        chart = Charts.Heatmap(
            dt=cm_out,
            PreAgg=True,
            XVar="predicted_name",
            YVar="actual_name",
            ValueVar="value",
            RenderHTML=plot_name,
            Theme=Theme,
            Title=title,
            SubTitle=subtitle,
            XAxisTitle="Predicted",
            YAxisTitle="Actual",
        )
    
        return {
            "table": cm_out,
            "plot": chart,
        }

    #################################################
    # Function: Partial Dependence Plots
    #################################################

    # Numeric Independent Variable Table
    def _build_pdp_numeric_table(
        self,
        feature: str,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        pred_col: str | None = None,
        n_bins: int = 20,
        binning: str = "quantile",  # "equal_width" or "quantile"
    ) -> pl.DataFrame:
        """
        Build a numeric partial dependence table for a single feature.
    
        Works for:
          - regression: target numeric, prediction column Predict_<target>
          - binary classification: target 0/1, probability column 'p1' (or custom)
    
        Returns a table with:
          feature_value (string for plotting), actual_mean, pred_mean, count
        """
    
        if binning not in ("equal_width", "quantile"):
            raise ValueError("binning must be 'equal_width' or 'quantile'.")
    
        # -------------------------
        # Resolve scored data
        # -------------------------
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to _build_pdp_numeric_table().")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )
    
        if feature not in df_pl.columns:
            raise ValueError(f"Feature '{feature}' not found in data.")
    
        # -------------------------
        # Resolve target & prediction
        # -------------------------
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError("self.TargetColumnName is None; supply target= for external df.")
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
    
        tt = self.TargetType.lower()
    
        if tt == "regression":
            if pred_col is None:
                pred_col = f"Predict_{target}"
        elif tt == "classification":
            # binary classification PDP uses p1 by default
            if pred_col is None:
                pred_col = "p1"
        else:
            raise NotImplementedError(
                "_build_pdp_numeric_table currently supports regression and binary classification only."
            )
    
        if pred_col not in df_pl.columns:
            raise ValueError(f"Prediction/probability column '{pred_col}' not found in data.")
    
        # -------------------------
        # Ensure feature is numeric
        # -------------------------
        dtype = df_pl.schema[feature]
        if not dtype.is_numeric():
            raise TypeError(
                f"PDP numeric: feature '{feature}' must be numeric, "
                f"but has dtype {dtype}"
            )
    
        # -------------------------
        # Assign bins on feature
        # -------------------------
        if binning == "equal_width":
            stats = df_pl.select(
                pl.col(feature).min().alias("min_val"),
                pl.col(feature).max().alias("max_val"),
            ).to_dicts()[0]
    
            min_val = float(stats["min_val"])
            max_val = float(stats["max_val"])
            if max_val == min_val:
                max_val = min_val + 1e-9
    
            width = max_val - min_val
    
            df_binned = df_pl.with_columns(
                (
                    ((pl.col(feature) - min_val) / width * n_bins)
                    .floor()
                    .cast(pl.Int64)
                    .clip(0, n_bins - 1)
                ).alias("__pdp_bin__")
            )
        else:  # "quantile"
            n_rows = df_pl.height
            if n_rows == 0:
                return pl.DataFrame([])
    
            df_binned = (
                df_pl
                .with_columns(
                    pl.col(feature)
                    .rank(method="average")
                    .alias("__rank__")
                )
                .with_columns(
                    (
                        ((pl.col("__rank__") - 1) / float(n_rows) * n_bins)
                        .floor()
                        .cast(pl.Int64)
                        .clip(0, n_bins - 1)
                    ).alias("__pdp_bin__")
                )
                .drop("__rank__")
            )
    
        # -------------------------
        # Aggregate per bin
        # -------------------------
        pdp_tbl = (
            df_binned
            .group_by("__pdp_bin__")
            .agg(
                pl.col(feature).mean().alias("feature_value"),
                pl.col(target).mean().alias("actual_mean"),
                pl.col(pred_col).mean().alias("pred_mean"),
                pl.len().alias("count"),
            )
            .sort("feature_value")
            .with_columns(
                # X-axis must be string for QuickEcharts; round for readability
                pl.col("feature_value")
                .round(2)
                .cast(pl.Utf8)
            )
            .select("feature_value", "actual_mean", "pred_mean", "count")
        )
    
        return pdp_tbl

    # Categorical Independent Variable Table
    def _build_pdp_categorical_table(
        self,
        feature: str,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        pred_col: str | None = None,
        sort_by: str = "feature",  # "feature", "actual_mean", or "pred_mean"
    ) -> pl.DataFrame:
        """
        Build a categorical partial dependence table for a single feature.
    
        For each category of `feature`, computes:
          - actual_mean
          - pred_mean (regression prediction or classification probability)
          - count
    
        Returns a table with columns: [feature, actual_mean, pred_mean, count].
        """
    
        # -------------------------
        # Resolve data
        # -------------------------
        if df is not None:
            df_pl = self._normalize_input_df(df)
        else:
            if DataName is None:
                raise ValueError("Must supply DataName or df to _build_pdp_categorical_table().")
            df_pl = self.ScoredData.get(DataName)
            if df_pl is None:
                raise ValueError(
                    f"self.ScoredData['{DataName}'] is None — run score() first."
                )
    
        if feature not in df_pl.columns:
            raise ValueError(f"Feature '{feature}' not found in data.")
    
        # -------------------------
        # Resolve target & prediction
        # -------------------------
        if target is None:
            if self.TargetColumnName is None:
                raise RuntimeError("self.TargetColumnName is None; supply target= for external df.")
            target = self.TargetColumnName
    
        if target not in df_pl.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
    
        tt = self.TargetType.lower()
    
        if tt == "regression":
            if pred_col is None:
                pred_col = f"Predict_{target}"
        elif tt == "classification":
            if pred_col is None:
                pred_col = "p1"
        else:
            raise NotImplementedError(
                "_build_pdp_categorical_table currently supports regression and binary classification only."
            )
    
        if pred_col not in df_pl.columns:
            raise ValueError(f"Prediction/probability column '{pred_col}' not found in data.")
    
        # -------------------------
        # Aggregate per category
        # -------------------------
        pdp_tbl = (
            df_pl
            .group_by(feature)
            .agg(
                pl.col(target).mean().alias("actual_mean"),
                pl.col(pred_col).mean().alias("pred_mean"),
                pl.len().alias("count"),
            )
        )
    
        # Sorting
        if sort_by == "actual_mean":
            pdp_tbl = pdp_tbl.sort("actual_mean")
        elif sort_by == "pred_mean":
            pdp_tbl = pdp_tbl.sort("pred_mean")
        else:  # "feature" or anything else
            pdp_tbl = pdp_tbl.sort(feature)
    
        return pdp_tbl

    # Numeric IV Partial Dependence Plot
    def plot_pdp_numeric(
        self,
        feature: str,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        pred_col: str | None = None,
        n_bins: int = 20,
        binning: str = "quantile",
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Numeric partial dependence plot for a single feature.
    
        Uses binned feature values on x-axis and:
          - actual_mean
          - pred_mean
        on y-axis (two-line overlay).
        """
        # Get table
        pdp_tbl = self._build_pdp_numeric_table(
            feature=feature,
            DataName=DataName if df is None else None,
            df=df,
            target=target,
            pred_col=pred_col,
            n_bins=n_bins,
            binning=binning,
        )

        data_label = DataName or "data"
        title = f"PDP (Numeric) · {feature} · {data_label}"

        # Build plot
        chart = Charts.Line(
            dt=pdp_tbl,
            PreAgg=True,
            YVar=["actual_mean", "pred_mean"],
            XVar="feature_value",
            RenderHTML=plot_name,
            Theme=Theme,
            Title=title,
            SubTitle=f"Bins = {n_bins}, binning = {binning}",
            YAxisTitle="Actual & Predicted",
            XAxisTitle=feature,
        )
    
        return {
            "table": pdp_tbl,
            "plot": chart,
        }

    # Categorical IV Partial Dependence Plot
    def plot_pdp_categorical(
        self,
        feature: str,
        DataName: str | None = "test",
        df=None,
        target: str | None = None,
        pred_col: str | None = None,
        sort_by: str = "feature",
        plot_name: str | None = None,
        Theme: str = "dark",
    ):
        """
        Categorical partial dependence plot for a single feature.
    
        X-axis: category levels of `feature`
        Y-axis: actual_mean and pred_mean
        """
        # Get table
        pdp_tbl = self._build_pdp_categorical_table(
            feature=feature,
            DataName=DataName if df is None else None,
            df=df,
            target=target,
            pred_col=pred_col,
            sort_by=sort_by,
        )
    
        data_label = DataName or "data"
        title = f"PDP (Categorical) · {feature} · {data_label}"

        # Build plot
        chart = Charts.Line(
            dt=pdp_tbl,
            PreAgg=True,
            YVar=["actual_mean", "pred_mean"],
            XVar=feature,
            RenderHTML=plot_name,
            Theme=Theme,
            Title=title,
            SubTitle=f"Sorted by {sort_by}",
            YAxisTitle="Actual & Predicted",
            XAxisTitle=feature,
        )
    
        return {
            "table": pdp_tbl,
            "plot": chart,
        }
