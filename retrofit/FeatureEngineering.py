# Module: FeatureEngineering
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.4
# Last modified : 2021-09-15


class FeatureEngineering:
    """
    Base class that the library specific classes inherit from.
    """

    def __init__(self) -> None:
        self.lag_args = {}
        self.roll_args = {}
        self.diff_args = {}
        self.calendar_args = {}
        self.dummy_args = {}
        self.partition_args = {}
        self._last_lag_args = {}
        self._last_roll_args = {}
        self._last_diff_args = {}
        self._last_calendar_args = {}
        self._last_dummy_args = {}
        self._last_partition_args = {}

    def save_args(self) -> None:
        self.lag_args = self._last_lag_args
        self.roll_args = self._last_roll_args
        self.diff_args = self._last_diff_args
        self.calendar_args = self._last_calendar_args
        self.dummy_args = self._last_dummy_args
        self.partition_args = self._last_partition_args

    def FE0_AutoLags(
        self,
        data=None,
        LagColumnNames=None,
        DateColumnName=None,
        ByVariables=None,
        LagPeriods=1,
        ImputeValue=-1,
        Sort=True,
        use_saved_args=False,
    ):
        raise NotImplementedError

    def FE0_AutoRollStats(
        data=None,
        ArgsList=None,
        RollColumnNames=None,
        DateColumnName=None,
        ByVariables=None,
        MovingAvg_Periods=None,
        MovingSD_Periods=None,
        MovingMin_Periods=None,
        MovingMax_Periods=None,
        ImputeValue=-1,
        Sort=True,
        use_saved_args=False,
    ):
        raise NotImplementedError

    def FE0_AutoDiff(
        data=None,
        ArgsList=None,
        DateColumnName=None,
        ByVariables=None,
        DiffNumericVariables=None,
        DiffDateVariables=None,
        DiffGroupVariables=None,
        NLag1=0,
        NLag2=1,
        Sort=True,
        use_saved_args=False,
    ):
        raise NotImplementedError
    
    def FE1_AutoCalendarVariables(
        data=None,
        ArgsList=None,
        DateColumnNames=None,
        CalendarVariables=None,
        use_saved_args=False
    ):
        raise NotImplementedError
      
    def FE1_DummyVariables(
        data=None,
        ArgsList=None,
        CategoricalColumnNames=None,
        use_saved_args=False
    ):
        raise NotImplementedError
    
    def FE2_AutoDataPartition(
        data = None, 
        ArgsList = None,
        DateColumnName = None,
        PartitionType = 'random',
        Ratios = None,
        ByVariables = None,
        Sort = False,
        use_saved_args = False
    ):
        raise NotImplementedError

