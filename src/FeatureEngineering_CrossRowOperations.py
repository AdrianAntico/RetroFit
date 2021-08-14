# Module: Cross Row Feature Engineering
# Author: Adrian Antico <adrianantico@gmail.com>
# License: Mozilla Public License 2.0
# Release: RetroFit.FeatureEngineering 0.0.1
# Last modified : 2021-08-11

def AutoLags(data = None, LagColumnNames = None, DateColumnName = None, ByVariables = None, LagPeriods = 1, ImputeValue = -1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate lags for multiple periods for multiple variables and by variables
    
    # Output
    Return datatable with new lag columns
    
    # Parameters
    data:           is your source datatable
    LagColumnNames: a list of columns that will be lagged
    DateColumnName: primary date column used for sorting
    ByVariables:    columns to lag by
    LagPeriods:              list of integers for the lookback lengths
    ImputeValue:    value to fill the NA's for beginning of series
    Sort:           sort the Frame before computing the lags - if you're data is sorted set this to False
    IntputFrame:    if you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    if you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test Function
    import datatable as dt
    from datatable import sort, f
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
    ## Group Example:
    data = AutoLags(data=data, LagPeriods=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
    print(data.names)
    
    ## Group and Multiple Periods and LagColumnNames:
    data = AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True)
    print(data.names)

    ## No Group Example:
    data = AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
    print(data.names)
    
    # QA: No Group Case: Step through function
    InputFrame='datatable'
    OutputFrame='datatable'
    LagPeriods = 1
    LagColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = None
    lp = 1
    ImputeValue = -1
    Sort = True
    
    # QA: Group Case: Step through function
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
    
    # Load minimal dependencies
    import datatable as dt
    from datatable import sort, f
    
    # Convert to datatable
    if InputFrame == 'pandas': 
      data = dt.Frame(data)

    # Ensure List
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]

    # Sort data
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName, reverse=True)]
    
    # Ensure List
    if not LagColumnNames is None and not isinstance(LagColumnNames, list):
      LagColumnNames = [LagColumnNames]

    # Ensure List
    if not LagPeriods is None and not isinstance(LagPeriods, list):
      LagPeriods = [LagPeriods]
    
    # Build lags
    for lcn in LagColumnNames:
      for lp in LagPeriods:
        Ref1 = "Lag_" + str(lp) + "_" + lcn
        if ByVariables is not None:
          data = data[:, f[:].extend({Ref1: dt.shift(f[lcn], n = lp)}), by(ByVariables)]
        else:
          data = data[:, f[:].extend({Ref1: dt.shift(f[lcn], n = lp)})]

    # Convert Frame
    if OutputFrame == 'pandas': data = data.to_pandas()
    
    # Return data
    return data


# Inner function for AutoRollStats
def RollStatSingleInstance(data, rcn, ns, ByVariables, ColsOriginal, MovingAvg_Periods_, MovingSD_Periods_, MovingMin_Periods_, MovingMax_Periods_):

  # Constants
  Ref = str(ns) + "_" + rcn
  Ref1 = "TEMP__Lag_" + Ref
  
  # Generate Lags for rowmean, rowsd, rowmin, rowmax
  if ByVariables is not None:
    data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = ns)}), by(ByVariables)]
  else:
    data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = ns)})]

  # Rolling Mean
  if ns in MovingAvg_Periods_:
    data = data[:, f[:].extend({"RollMean_" + Ref: dt.rowmean(f[Ref1])})]
    
  # Rolling SD
  if ns in MovingSD_Periods_:
    data = data[:, f[:].extend({"RollSD_" + Ref: dt.rowsd(f[Ref1])})]
    
  # Rolling Min
  if ns in MovingMin_Periods_:
    data = data[:, f[:].extend({"RollMin_" + Ref: dt.rowmin(f[Ref1])})]
    
  # Rolling Max
  if ns in MovingMax_Periods_:
    data = data[:, f[:].extend({"RollMax_" + Ref: dt.rowmax(f[Ref1])})]
    
  # Return
  return data


def AutoRollStats(data = None, RollColumnNames = None, DateColumnName = None, ByVariables = None, MovingAvg_Periods = 2, MovingSD_Periods = None, MovingMin_Periods = None, MovingMax_Periods = None, ImputeValue = -1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
    # Output
    Return datatable with new rolling statistics columns
    
    # Parameters
    data:             Source data
    RollColumnNames:  A list of columns that will be lagged
    DateColumnName:   Primary date column used for sorting
    ByVariables:      Columns to lag by
    Moving_*_Periods: List of integers for look back window
    ImputeValue:      Value to fill the NA's for beginning of series
    Sort:             Sort the Frame before computing the lags - if you're data is sorted set this to False
    IntputFrame:      'datatable' or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:      'datatable' or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test Function
    
    ## Group Example:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    print(data.names)
    
    ## Group and Multiple Periods and RollColumnNames:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    print(data.names)

    ## No Group Example:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    print(data.names)
    
    # QA: No Group Case: Step through function
    InputFrame='datatable'
    OutputFrame='datatable'
    MovingAvg_Periods = 2
    MovingSD_Periods = 2
    MovingMin_Periods = 2
    MovingMax_Periods = 2
    RollColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = None
    ns = 1
    ImputeValue = -1
    Sort = True
    
    # QA: Group Case: Step through function
    InputFrame='datatable'
    OutputFrame='datatable'
    MovingAvg_Periods = 2
    MovingSD_Periods = 2
    MovingMin_Periods = 2
    MovingMax_Periods = 2
    RollColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    ns = 1
    ImputeValue = -1
    Sort = True
    """
    
    # Load minimal dependencies
    import datatable as dt
    from datatable import sort, f, by
    
    # Convert to datatable
    if InputFrame == 'pandas': 
      data = dt.Frame(data)

    # Ensure ByVariables is a list
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]

    # Sort data if requested
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName)]

    # Prepare column and value references
    if not RollColumnNames is None and not isinstance(RollColumnNames, list):
      RollColumnNames = [RollColumnNames]

    # Ensure List
    if not MovingAvg_Periods is None and not isinstance(MovingAvg_Periods, list):
      MovingAvg_Periods = [MovingAvg_Periods]

    # Ensure List
    if not MovingSD_Periods is None and not isinstance(MovingSD_Periods, list):
      MovingSD_Periods = [MovingSD_Periods]

    # Ensure List
    if not MovingMin_Periods is None and not isinstance(MovingMin_Periods, list):
      MovingMin_Periods = [MovingMin_Periods]

    # Ensure List
    if not MovingMax_Periods is None and not isinstance(MovingMax_Periods, list):
      MovingMax_Periods = [MovingMax_Periods]

    # Build lags to max window value
    ColsOriginal = data.names
    MaxVal = max(MovingAvg_Periods, MovingSD_Periods, MovingMin_Periods, MovingMax_Periods)[0]
    for rcn in RollColumnNames:
      for ns in range(1, MaxVal+1):
        data = RollStatSingleInstance(data, rcn, ns, ByVariables, ColsOriginal, MovingAvg_Periods_=MovingAvg_Periods, MovingSD_Periods_=MovingSD_Periods, MovingMin_Periods_=MovingMin_Periods, MovingMax_Periods_=MovingMax_Periods)

      # Remove Temporary Lagged Columns
      del data[:, [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]]

    # Convert Frame
    if OutputFrame == 'pandas': data = data.to_pandas()
    
    # Return data
    return data


def AutoDiff(data = None, DateColumnName = None, ByVariables = None, DiffNumericVariables = None, DiffDateVariables = None, DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
    # Output
    Return datatable with new rolling statistics columns
    
    # Parameters
    data:                 Source data
    DateColumnName:       Primary date column used for sorting
    ByVariables:          By grouping variables
    DiffNumericVariables: None
    DiffDateVariables:    None
    DiffGroupVariables:   None
    NLag1:                Default 0. 0 means the current value - NLag2_Current_Value
    NLag2:                Default 1. 1 means a lag1 of the current value
    Sort:                 True or False
    InputFrame:           'datatable' or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:          'datatable' or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test Function
    
    ## Group Example:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    print(data.names)
    
    ## Group and Multiple Periods and RollColumnNames:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    print(data.names)

    ## No Group Example:
    import datatable as dt
    from datatable import sort, f, by
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    print(data.names)
    
    # QA: No Group Case: Step through function
    data=data
    DateColumnName = 'CalendarDateColumn'
    ByVariables = None
    DiffNumericVariables = 'Leads'
    DiffDateVariables = 'CalendarDateColumn'
    DiffGroupVariables = None
    NLag1 = 0
    NLag2 = 1
    Sort=True
    InputFrame = 'datatable'
    OutputFrame = 'datatable'
    rcn = 'Leads'

    # QA: Group Case: Step through function
    data=data
    DateColumnName = 'CalendarDateColumn'
    ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    DiffNumericVariables = 'Leads'
    DiffDateVariables = 'CalendarDateColumn'
    DiffGroupVariables = None
    NLag1 = 0
    NLag2 = 1
    Sort=True
    InputFrame = 'datatable'
    OutputFrame = 'datatable'
    rcn = 'Leads'
    """
    
    # Load minimal dependencies
    import datatable as dt
    from datatable import sort, f, by
    
    # Convert to datatable
    if InputFrame == 'pandas': 
      data = dt.Frame(data)

    # Ensure ByVariables is a list
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]

    # Ensure DiffNumericVariables is a list
    if not DiffNumericVariables is None and not isinstance(DiffNumericVariables, list):
      DiffNumericVariables = [DiffNumericVariables]

    # Ensure DiffDateVariables is a list
    if not DiffDateVariables is None and not isinstance(DiffDateVariables, list):
      DiffDateVariables = [DiffDateVariables]

    # Ensure DiffGroupVariables is a list
    if not DiffGroupVariables is None and not isinstance(DiffGroupVariables, list):
      DiffGroupVariables = [DiffGroupVariables]

    # Sort data if requested
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName)]

    # DiffNumericVariables
    if not DiffNumericVariables is None:
      for rcn in DiffNumericVariables:
        
        # Numeric Variable Procedure
        if NLag1 == 0:
          
          # Create Lags
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]

          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: f[rcn] - f[Ref2]})]

          # Remove temp columns
          del data[:, f[Ref2]]

        else:
          
          # Create Lags
          Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)}), by(ByVariables)]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)})]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]
          
          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: f[Ref1] - f[Ref2]})]
          
          # Remove temp columns
          del data[:, f[Ref1]]
          del data[:, f[Ref2]]

    # DiffDateVariables
    if not DiffDateVariables is None:
      for rcn in DiffDateVariables:

        # Date Variable Procedure
        if NLag1 == 0:
          
          # Create Lags
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]

          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: dt.as_type(f[rcn], int) - dt.as_type(f[Ref2], int)})]
          
          # Remove temp columns
          del data[:, f[Ref2]]

        else:
          
          # Create Lags
          Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)}), by(ByVariables)]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)})]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]
          
          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: dt.as_type(f[rcn], int) - dt.as_type(f[Ref2], int)})]
          
          # Remove temp columns
          del data[:, f[Ref1]]
          del data[:, f[Ref2]]

    # DiffGroupVariables
    if not DiffGroupVariables is None:
      for rcn in DiffGroupVariables:
        
        # Date Variable Procedure
        if NLag1 == 0:
          
          # Create Lags
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]

          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: dt.ifelse(f[rcn] == f[Ref2], "NoDiff", "New=" + f[rcn] + "Old=" + f[Ref2])})]
          
          # Remove temp columns
          del data[:, f[Ref2]]

        else:
          
          # Create Lags
          Ref1 = "TEMP__Lag_" + str(NLag1) + "_" + rcn
          Ref2 = "TEMP__Lag_" + str(NLag2) + "_" + rcn
          if not ByVariables is None:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)}), by(ByVariables)]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = NLag1)})]
            data = data[:, f[:].extend({Ref2: dt.shift(f[rcn], n = NLag2)})]
          
          # Create diffs
          data = data[:, f[:].extend({"Diff_" + str(NLag1) + "_" + str(NLag2) + "_" + rcn: dt.ifelse(f[rcn] == f[Ref2], "NoDiff", "New=" + f[rcn] + "Old=" + f[Ref2])})]
          
          # Remove temp columns
          del data[:, f[Ref1]]
          del data[:, f[Ref2]]

    # Convert Frame
    if OutputFrame == 'pandas': data = data.to_pandas()
    
    # Return data
    return data
