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

    # Sort data if requested
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName)]
    
    # Prepare column and value references
    if not LagColumnNames is None and not isinstance(LagColumnNames, list):
      LagColumnNames = [LagColumnNames]
    if not LagPeriods is None and not isinstance(LagPeriods, list):
      LagPeriods = [LagPeriods]
    
    # Build lags
    Cols = data.names
    for lcn in LagColumnNames:
      colnum = Cols.index(lcn)
      for lp in LagPeriods:
        if ByVariables is not None:
          data = data[:, f[:].extend({"Lag_" + str(lp) + "_" + lcn: dt.shift(f[colnum], n = lp)}), by(ByVariables)]
        else:
          data = data[:, f[:].extend({"Lag_" + str(lp) + "_" + lcn: dt.shift(f[colnum], n = lp)})]

    # Convert Frame
    if OutputFrame == 'pandas': data = data.to_pandas()
    
    # Return data
    return data


# Inner function for AutoRollStats
def RollStatSingleInstance(data, rcn, ns, ByVariables, ColsOriginal, MovingAvg_Periods_, MovingSD_Periods_, MovingMin_Periods_, MovingMax_Periods_):

  # Metadata for column number identifiers
  Cols = data.names
  colnum = Cols.index(rcn)

  # Generate Lags for rowmean, rowsd, rowmin, rowmax
  if ByVariables is not None:
    data = data[:, f[:].extend({"TEMP__Lag_" + str(ns) + "_" + rcn: dt.shift(f[colnum], n = ns)}), by(ByVariables)]
  else:
    data = data[:, f[:].extend({"TEMP__Lag_" + str(ns) + "_" + rcn: dt.shift(f[colnum], n = ns)})]

  # Metadata
  MA_Cols = list(set(data.names) - set(ColsOriginal))
  colnumtemp = [data.names.index(MA_Cols[gg-1]) for gg in range(1,len(MA_Cols)+1)]

  # Rolling Mean
  if ns in MovingAvg_Periods_:
    data = data[:, f[:].extend({"RollMean_" + str(ns) + "_" + rcn: dt.rowmean(f[colnumtemp])})]
    
  # Rolling SD
  if ns in MovingSD_Periods_:
    data = data[:, f[:].extend({"RollSD_" + str(ns) + "_" + rcn: dt.rowsd(f[colnumtemp])})]
    
  # Rolling Min
  if ns in MovingMin_Periods_:
    data = data[:, f[:].extend({"RollMin_" + str(ns) + "_" + rcn: dt.rowmin(f[colnumtemp])})]
    
  # Rolling Max
  if ns in MovingMax_Periods_:
    data = data[:, f[:].extend({"RollMax_" + str(ns) + "_" + rcn: dt.rowmax(f[colnumtemp])})]
    
  # Return
  return data


def AutoRollStats(data = None, RollColumnNames = None, DateColumnName = None, ByVariables = None, MovingAvg_Periods = 2, MovingSD_Periods = None, MovingMin_Periods = None, MovingMax_Periods = None, ImputeValue = -1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
    # Output
    Return datatable with new rolling statistics columns
    
    # Parameters
    data:             is your source datatable
    RollColumnNames:   a list of columns that will be lagged
    DateColumnName:   primary date column used for sorting
    ByVariables:      columns to lag by
    Moving_*_Periods: list of integers for look back window
    ImputeValue:      value to fill the NA's for beginning of series
    Sort:             sort the Frame before computing the lags - if you're data is sorted set this to False
    IntputFrame:      if you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:      if you want the output Frame to be pandas change value to 'pandas'
    
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
