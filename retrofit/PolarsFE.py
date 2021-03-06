# Module: PolarsFE
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

import copy
import numpy as np
import polars as pl
from polars import col, when
from polars.lazy import col
from retrofit.FeatureEngineering import FeatureEngineering
from retrofit import utils as u


class FE(FeatureEngineering):
    def __init__(self) -> None:
        super().__init__()

    def FE0_AutoLags(
        self,
        data=None,
        LagColumnNames=None,
        DateColumnName=None,
        ByVariables=None,
        LagPeriods=1,
        ImputeValue=-1,
        Sort=True,
        use_saved_args=False):
        """
        # TODO:  Update doc string and examples.  Only use Datatable in examples.
        # Goal:
        Automatically generate lags for multiple periods for multiple variables and by variables

        # Output
        Return a datatable, polars frame, or pandas frame with new lag columns

        # Parameters
        data:           Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        LagColumnNames: A list of columns that will be lagged
        DateColumnName: Primary date column used for sorting
        ByVariables:    Columns to partition over
        LagPeriods:     List of integers for the lookback lengths
        ImputeValue:    Value to fill the NA's for beginning of series
        Sort:           Sort the Frame before computing the lags - if you're data is sorted set this to False
        use_saved_args: Score mode
        """

        # ArgsList Collection
        if use_saved_args:
            LagColumnNames = self.lag_args.get("LagColumnNames")
            DateColumnName = self.lag_args.get("DateColumnName")
            ByVariables = self.lag_args.get("ByVariables")
            LagPeriods = self.lag_args.get("LagPeriods")
            ImputeValue = self.lag_args.get("ImputeValue")

        # Locals is a dict of args and their respective values
        self._last_lag_args = locals()

        # Ensure List
        if isinstance(LagColumnNames, str):
            LagColumnNames = [LagColumnNames]
        elif not isinstance(LagColumnNames, (list, type(None))):
            raise Exception("LagColumnNames should be a string or a list")

        if isinstance(LagPeriods, int):
            LagPeriods = [LagPeriods]
        elif not isinstance(LagPeriods, (list, type(None))):
            raise Exception("LagPeriods should be a string or a list")

        if isinstance(ByVariables, str):
            ByVariables = [ByVariables]
        elif not isinstance(ByVariables, (list, type(None))):
            raise Exception("ByVariables should be a string or a list")

        if isinstance(DateColumnName, str):
            DateColumnName = [DateColumnName]
        elif not isinstance(DateColumnName, (list, type(None))):
            raise Exception("DateColumnName should be a string or a list")

        # Sort data
        if ByVariables:
            SortCols = copy.copy(ByVariables)
            SortCols.extend(DateColumnName)
            rev = [True for t in range(len(SortCols))]
            data.sort(SortCols, reverse = rev, in_place = True)
        else:
            if not isinstance(data[DateColumnName].dtype(), pl.Date32):
                data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
            data.sort(DateColumnName, reverse = True, in_place = True)

        # Create lags
        for lcn in LagColumnNames:
            for lp in LagPeriods:
              
                # New Column Name
                Ref1 = "Lag_" + str(lp) + "_" + lcn
                
                # Generate lags
                if ByVariables:
                    if ImputeValue:
                      data = (data.select([
                          pl.all(),
                          col(lcn).shift_and_fill(lp, ImputeValue).over(ByVariables).explode().alias(Ref1)]))
                    else:
                      data = (data.select([
                          pl.all(),
                          col(lcn).shift(lp).over(ByVariables).explode().alias(Ref1)]))
                else:
                    if ImputeValue:
                        data = (data.select([
                          pl.all(),
                          col(lcn).shift_and_fill(lp, ImputeValue).alias(Ref1)]))
                    else:
                        data = (data.select([
                          pl.all(),
                          col(lcn).shift(lp).alias(Ref1)]))

        return data

    # in class
    def FE0_AutoRollStats(
        self,
        data=None,
        RollColumnNames=None,
        DateColumnName=None,
        ByVariables=None,
        MovingAvg_Periods=None,
        MovingSD_Periods=None,
        MovingMin_Periods=None,
        MovingMax_Periods=None,
        ImputeValue=-1,
        Sort=True,
        use_saved_args=False):
        """
        # TODO: Update doc strings and examples to only use datatable.
        # Goal:
        Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
        # Output
        Return a datatable, polars frame, or pandas frame with new rolling statistics columns
    
        # Parameters
        data:             Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        RollColumnNames:  A list of columns that will be lagged
        DateColumnName:   Primary date column used for sorting
        ByVariables:      Columns to partition over
        Moving*_Periods:  List of integers for look back window
        ImputeValue:      Value to fill the NA's for beginning of series
        Sort:             Sort the Frame before computing the lags - if you're data is sorted set this to False
        use_saved_args:   Score mode
        """
    
        # ArgsList Collection
        if use_saved_args:
            RollColumnNames = self.roll_args.get("RollColumnNames")
            DateColumnName = self.roll_args.get("DateColumnName")
            ByVariables = self.roll_args.get("ByVariables")
            MovingAvg_Periods = self.roll_args.get("MovingAvg_Periods")
            MovingSD_Periods = self.roll_args.get("MovingSD_Periods")
            MovingMin_Periods = self.roll_args.get("MovingMin_Periods")
            MovingMax_Periods = self.roll_args.get("MovingMax_Periods")
            ImputeValue = self.roll_args.get("ImputeValue")
    
        self._last_roll_args = locals()
    
        # Ensure List
        if isinstance(DateColumnName, str):
            DateColumnName = [DateColumnName]
        elif not isinstance(DateColumnName, (list, type(None))):
            raise Exception("DateColumnName should be a string or a list")

        if isinstance(ByVariables, str):
            ByVariables = [ByVariables]
        elif not isinstance(ByVariables, (list, type(None))):
            raise Exception("ByVariables should be a string or a list")
    
        if isinstance(RollColumnNames, str):
            RollColumnNames = [RollColumnNames]
        elif not isinstance(ByVariables, (list, type(None))):
            raise Exception("RollColumnNames should be a string or a list")
    
        if isinstance(MovingAvg_Periods, int):
            MovingAvg_Periods = [MovingAvg_Periods]
        elif not isinstance(MovingAvg_Periods, (list, type(None))):
            raise Exception("MovingAvg_Periods should be a string or a list")
    
        if isinstance(MovingSD_Periods, int):
            MovingSD_Periods = [MovingSD_Periods]
        elif not isinstance(MovingSD_Periods, (list, type(None))):
            raise Exception("MovingSD_Periods should be a string or a list")
    
        if isinstance(MovingMin_Periods, int):
            MovingMin_Periods = [MovingMin_Periods]
        elif not isinstance(MovingMin_Periods, (list, type(None))):
            raise Exception("MovingMin_Periods should be a string or a list")
    
        if isinstance(MovingMax_Periods, int):
            MovingMax_Periods = [MovingMax_Periods]
        elif not isinstance(MovingMax_Periods, (list, type(None))):
            raise Exception("MovingMax_Periods should be a string or a list")
    
        # Sort data
        if ByVariables:
            SortCols = copy.copy(ByVariables)
            SortCols.extend(DateColumnName)
            rev = [True for t in range(len(SortCols))]
            data.sort(SortCols, reverse = rev, in_place = True)
        else:
            if not isinstance(data[DateColumnName].dtype(), pl.Date32):
                data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
            data.sort(DateColumnName, reverse = True, in_place = True)
    
        # Build lags to max window value
        MaxVal = max(max(MovingAvg_Periods, MovingSD_Periods, MovingMin_Periods, MovingMax_Periods))
    
        # processing
        for rcn in RollColumnNames:
            for ns in range(1, MaxVal + 1):
    
                # Constants
                Ref = str(ns) + "_" + rcn
                Ref1 = "TEMP__Lag_" + Ref
    
                # Generate Lags for rowmean, rowsd, rowmin, rowmax
                if ByVariables:
                    raise Exception("No method yet")
                else:
                    raise Exception("No method yet")
    
                # Rolling Mean
                if ns in MovingAvg_Periods:
                    Ref2 = [zzz for zzz in data.names if "TEMP__Lag_" in zzz]
                    raise Exception("No method yet")
    
                # Rolling SD
                if ns in MovingSD_Periods:
                    Ref2 = [zzz for zzz in data.names if "TEMP__Lag_" in zzz]
                    raise Exception("No method yet")
    
                # Rolling Min
                if ns in MovingMin_Periods:
                    Ref2 = [zzz for zzz in data.names if "TEMP__Lag_" in zzz]
                    raise Exception("No method yet")
    
                # Rolling Max
                if ns in MovingMax_Periods:
                    Ref2 = [zzz for zzz in data.names if "TEMP__Lag_" in zzz]
                    raise Exception("No method yet")
    
            # Remove Temporary Lagged Columns
            raise Exception("No method yet")
    
        # Return data
        return data

    # in class
    def FE0_AutoDiff(
        self,
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
        use_saved_args=False):
    
        """
        # Goal:
        Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
        # Output
        Return a datatable, polars frame, or pandas frame with new difference columns
    
        # Parameters
        data:                 Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        DateColumnName:       Primary date column used for sorting
        ByVariables:          Columns to partition over
        DiffNumericVariables: Numeric variable name scalar or list
        DiffDateVariables:    Date variable name scalar or list
        DiffGroupVariables:   Categorical variable name scalar or list
        NLag1:                Default 0. 0 means the current value - NLag2_Current_Value, otherwise NLag1_Current_Value - NLag2_Current_Value
        NLag2:                Default 1. 1 means a lag1 of the current value
        Sort:                 True or False
        use_saved_args:       Scoring mode
        """
        
        # ArgsList Collection
        if use_saved_args:
            DateColumnName = self.diff_args.get("DateColumnName")
            ByVariables = self.diff_args.get("ByVariables")
            DiffNumericVariables = self.diff_args.get('DiffNumericVariables')
            DiffDateVariables = self.diff_args.get('DiffDateVariables')
            DiffGroupVariables = self.diff_args.get('DiffGroupVariables')
            NLag1 = self.diff_args.get('NLag1')
            NLag2 = self.diff_args.get('NLag2')
            
        # Locals is a dict of args and their respective values
        self._last_lag_args = locals()
            
        # Ensure List
        if isinstance(DateColumnName, str):
            DateColumnName = [DateColumnName]
        elif not isinstance(DateColumnName, (list, type(None))):
            raise Exception("DateColumnName should be a string or a list")

        if isinstance(ByVariables, str):
            ByVariables = [ByVariables]
        elif not isinstance(ByVariables, (list, type(None))):
            raise Exception("ByVariables should be a string or a list")
    
        if isinstance(DiffNumericVariables, str):
            DiffNumericVariables = [DiffNumericVariables]
        elif not isinstance(DiffNumericVariables, (list, type(None))):
            raise Exception("DiffNumericVariables should be a string or a list")
    
        if isinstance(DiffDateVariables, str):
            DiffDateVariables = [DiffDateVariables]
        elif not isinstance(DiffDateVariables, (list, type(None))):
            raise Exception("DiffDateVariables should be a string or a list")
    
        if isinstance(DiffGroupVariables, str):
            DiffGroupVariables = [DiffGroupVariables]
        elif not isinstance(DiffGroupVariables, (list, type(None))):
            raise Exception("DiffGroupVariables should be a string or a list")

        # Sort data
        if ByVariables:
            SortCols = copy.copy(ByVariables)
            SortCols.extend(DateColumnName)
            rev = [True for t in range(len(SortCols))]
            data.sort(SortCols, reverse = rev, in_place = True)
        else:
            if not isinstance(data[DateColumnName].dtype(), pl.Date32):
                data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
            data.sort(DateColumnName, reverse = True, in_place = True)

        # DiffNumericVariables
        if DiffNumericVariables:
            for rcn in DiffNumericVariables:

                # Numeric Variable Procedure
                if NLag1 == 0:

                    # Create Lags
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}), by(ByVariables)]
                    else:
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: f[rcn] - f[Ref2]})]

                    # Remove temp columns
                    del data[:, f[Ref2]]

                else:

                    # Create Lags
                    Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[
                            :,
                            f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)}),
                            by(ByVariables),
                        ]
                        data = data[
                            :,
                            f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}),
                            by(ByVariables),
                        ]
                    else:
                        data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)})]
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[
                        :,
                        f[:].extend(
                            {
                                "Diff_"
                                + str(NLag1)
                                + "_"
                                + str(NLag2)
                                + "_"
                                + rcn: f[Ref1]
                                - f[Ref2]
                            }
                        ),
                    ]

                    # Remove temp columns
                    del data[:, f[Ref1]]
                    del data[:, f[Ref2]]

        # DiffDateVariables
        if DiffDateVariables:
            for rcn in DiffDateVariables:

                # Date Variable Procedure
                if NLag1 == 0:

                    # Create Lags
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[
                            :,
                            f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}),
                            by(ByVariables),
                        ]
                    else:
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[
                        :,
                        f[:].extend(
                            {
                                "Diff_"
                                + str(NLag1)
                                + "_"
                                + str(NLag2)
                                + "_"
                                + rcn: dt.as_type(f[rcn], int)
                                - dt.as_type(f[Ref2], int)
                            }
                        ),
                    ]

                    # Remove temp columns
                    del data[:, f[Ref2]]

                else:

                    # Create Lags
                    Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[
                            :,
                            f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)}),
                            by(ByVariables),
                        ]
                        data = data[
                            :,
                            f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}),
                            by(ByVariables),
                        ]
                    else:
                        data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)})]
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[
                        :,
                        f[:].extend(
                            {
                                "Diff_"
                                + str(NLag1)
                                + "_"
                                + str(NLag2)
                                + "_"
                                + rcn: dt.as_type(f[rcn], int)
                                - dt.as_type(f[Ref2], int)
                            }
                        ),
                    ]

                    # Remove temp columns
                    del data[:, f[Ref1]]
                    del data[:, f[Ref2]]

        # DiffGroupVariables
        if DiffGroupVariables:
            for rcn in DiffGroupVariables:

                # Date Variable Procedure
                if NLag1 == 0:

                    # Create Lags
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[
                            :,
                            f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}),
                            by(ByVariables),
                        ]
                    else:
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[
                        :,
                        f[:].extend(
                            {
                                "Diff_"
                                + str(NLag1)
                                + "_"
                                + str(NLag2)
                                + "_"
                                + rcn: dt.ifelse(
                                    f[rcn] == f[Ref2],
                                    "NoDiff",
                                    "New=" + f[rcn] + "Old=" + f[Ref2],
                                )
                            }
                        ),
                    ]

                    # Remove temp columns
                    del data[:, f[Ref2]]

                else:

                    # Create Lags
                    Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
                    Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
                    if ByVariables:
                        data = data[
                            :,
                            f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)}),
                            by(ByVariables),
                        ]
                        data = data[
                            :,
                            f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)}),
                            by(ByVariables),
                        ]
                    else:
                        data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n=NLag1)})]
                        data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n=NLag2)})]

                    # Create diffs
                    data = data[
                        :,
                        f[:].extend(
                            {
                                "Diff_"
                                + str(NLag1)
                                + "_"
                                + str(NLag2)
                                + "_"
                                + rcn: dt.ifelse(
                                    f[rcn] == f[Ref2],
                                    "NoDiff",
                                    "New=" + f[rcn] + "Old=" + f[Ref2],
                                )
                            }
                        ),
                    ]

                    # Remove temp columns
                    del data[:, f[Ref1]]
                    del data[:, f[Ref2]]

        # Return data
        return data


    # in class
    def FE1_AutoCalendarVariables(
        self,
        data=None,
        DateColumnNames=None,
        CalendarVariables=None,
        use_saved_args=False):
    
        """
        # Goal:
        Automatically generate calendar variables from your date columns
    
        # Output
        Return a datatable, polars, or pandas frame with new calendar variables
    
        # Parameters
        data:                 Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        DateColumnNames:      Primary date column used for sorting
        CalendarVariables:    'nanosecond', 'second', 'minute', 'hour', 'mday', 'wday', 'month', 'quarter', 'year'
        use_saved_args:       Score mode
        """
        # Imports
        from polars.eager.series import DateTimeNameSpace as dts
        
        # ArgsList Collection
        if use_saved_args:
            DateColumnNames = self.calendar_args.get("DateColumnName")
            CalendarVariables = self.calendar_args.get("CalendarVariables")

        # Locals is a dict of args and their respective values
        self._last_calendar_args = locals()
            
        # Ensure List
        if isinstance(DateColumnNames, str):
            DateColumnNames = [DateColumnNames]
        elif not isinstance(DateColumnNames, (list, type(None))):
            raise Exception("DateColumnNames should be a string or a list")
        
        # Ensure List
        if isinstance(CalendarVariables, str):
            CalendarVariables = [CalendarVariables]
        elif not isinstance(CalendarVariables, (list, type(None))):
            raise Exception("CalendarVariables should be a string or a list")

        # Ensure DateColumn is a Date32 or Date64
        if isinstance(data[DateVar], pl.Date32) or isinstance(data[DateVar], pl.Date64):
            data[DateVar] = data[DateVar].cast(pl.Date32)
        
        # Loop through DateColumns
        for DateVar in DateColumnNames:
            for CVars in CalendarVariables:
    
                # Nanosecond
                if CVars.lower() in "nanosecond":
                    try:
                        data[f"{DateVar}_nanosecond"] = dts.nanosecond(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.nanosecond calculation due to type mismatch")
    
                # Second
                if CVars.lower() in "second":
                    try:
                        data[f"{DateVar}_second"] = dts.second(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.second calculation due to type mismatch")
    
                # Minute
                if CVars.lower() in "minute":
                    try:
                        data[f"{DateVar}_minute"] = dts.minute(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.minute calculation due to type mismatch")
    
                # Hour
                if CVars.lower() in "hour":
                    try:
                        data[f"{DateVar}_hour"] = dts.hour(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.hour calculation due to type mismatch")
    
                # day_of_week
                if CVars.lower() in "wday":
                    try:
                        data[f"{DateVar}_wday"] = dts.weekday(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.day_of_week 'wday' calculation due to type mismatch")
    
                # day of month
                if CVars.lower() in "mday":
                    try:
                        data[f"{DateVar}_mday"] = dts.day(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.day 'mday' calculation due to type mismatch")
    
                # month
                if CVars.lower() in "month":
                    try:
                        data[f"{DateVar}_month"] = dts.month(data[DateVar])
                    except ValueError:
                        raise print("Skipping wday time.month calculation due to type mismatch")
    
                # quarter
                if CVars.lower() in "quarter":
                    try:
                        data["temp___temp"] = dts.month(data[DateVar])
                        temp = pl.DataFrame({"temp___temp": [1,2,3,4,5,6,7,8,9,10,11,12], f"{DateVar}_quarter": [1,1,1,2,2,2,3,3,3,4,4,4]})
                        temp['temp___temp'] = temp['temp___temp'].cast(pl.UInt32)
                        data = data.join(temp, on = "temp___temp")
                        data = data.drop('temp___temp')
                    except ValueError:
                        raise print("Skipping time.month 'quarter' calculation due to type mismatch")
    
                # year
                if CVars.lower() in "year":
                    try:
                        data[f"{DateVar}_year"] = dts.year(data[DateVar])
                    except ValueError:
                        raise print("Skipping time.year calculation due to type mismatch")
    
        # Return
        return data


    def FE1_DummyVariables(
        self,
        data=None,
        CategoricalColumnNames=None,
        use_saved_args=False):
        """
        # Goal:
        Automatically generate dummy variables for CategoricalColumnNames provided by user
    
        # Output
        Return a datatable
    
        # Parameters
        data:                   Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        CategoricalColumnNames: Scalar. Primary date column used for sorting
        use_saved_args:         Scoring mode
        """
        
        # ArgsList Collection
        if use_saved_args:
            CategoricalColumnNames = self.calendar_args.get("CategoricalColumnNames")

        # Locals is a dict of args and their respective values
        self._last_calendar_args = locals()
            
        # Ensure List
        if isinstance(CategoricalColumnNames, str):
            CategoricalColumnNames = [CategoricalColumnNames]
        elif not isinstance(CategoricalColumnNames, (list, type(None))):
            raise Exception("CategoricalColumnNames should be a string or a list")

        # Create dummies
        for column in CategoricalColumnNames:
            data = data.hstack(pl.get_dummies(data[column]))

        # Return data
        return data_new


    # in class
    def FE2_AutoDataParition(
        self,
        data=None,
        DateColumnName=None,
        PartitionType="random",
        Ratios=None,
        ByVariables=None,
        Sort=False,
        use_saved_args=False):
    
        """
        # Goal:
        Automatically generate train, validation, and test data sets for modeling purposes
    
        # Output
        Returns paritioned data sets for ML
    
        # Parameters
        data:           Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
        DateColumnName: Scalar. Primary date column used for sorting
        PartitionType:  Scalar. Columns to partition over
        Ratios:         List. Use ths for PartitionType 'random'. List of decimal values for determining how many data goes into each data frame.
        Sort:           Sort data before creating time based partitions
        ByVariables:    None or List. Stratify the data paritioning using ByVariables
        use_saved_args: Score mode
        """

        # ArgsList Collection
        if use_saved_args:
            DateColumnName = self.partition_args.get("DateColumnName")
            PartitionType = self.partition_args.get("PartitionType")
            Ratios = self.partition_args.get("Ratios")
            ByVariables = self.partition_args.get("ByVariables")

        # Locals is a dict of args and their respective values
        self._last_calendar_args = locals()
            
        # Ensure List
        if isinstance(CategoricalColumnNames, str):
            CategoricalColumnNames = [CategoricalColumnNames]
        elif not isinstance(CategoricalColumnNames, (list, type(None))):
            raise Exception("CategoricalColumnNames should be a string or a list")

        # Accumulate Ratios
        Ratios = u.cumsum(Ratios)
    
        # Random partitioning
        if PartitionType.lower() == 'random':
          
            # Prepare data
            data['ID'] = np.random.uniform(0,1, size = data.shape[0])
            data = data.sort('ID')
            data.drop_in_place('ID')
        
        # Time base partitioning
        if PartitionType.lower() == "time":
          
            # Prepare data
            if Sort:
                data.sort(DateColumnName, reverse = False, in_place = True)
        
        # Number of rows
        NumRows = data.shape[0]
          
        # Grab row number boundaries
        TrainRowsMax = NumRows * Ratios[0]
        ValidRowsMax = NumRows * Ratios[1]
        
        # TrainData
        TrainData = data[:int(TrainRowsMax)]
  
        # ValidationData
        ValidationData = data[int(TrainRowsMax + 1):int(ValidRowsMax)]
        
        # TestData
        if len(Ratios) == 3:
            TestData = data[int(ValidRowsMax + 1):]
        else:
            TestData = None
    
        # Return data
        return dict(TrainData=TrainData, ValidationData=ValidationData, TestData=TestData)
