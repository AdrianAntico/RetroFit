# Module: PandasFE
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.4
# Last modified : 2021-09-15


import pandas as pd
from retrofit.FeatureEngineering import FeatureEngineering


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
        use_saved_args=False,
    ):
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
        self._last_lag_args.pop("data")

        # Build lags
        for lcn in LagColumnNames:
            for lp in LagPeriods:
                # New Column Name
                Ref1 = f"Lag_{lp}_{lcn}"
                # Generate lags
                if ByVariables:
                    data[Ref1] = data.groupby(ByVariables)[lcn].shift(periods=lp,)
                else:
                    data[Ref1] = data.loc[:, lcn].shift(lp, axis=1)
                if ImputeValue:
                    data[Ref1].fillna(ImputeValue, inplace=True)

        return data

