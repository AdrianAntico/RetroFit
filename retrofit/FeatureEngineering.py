# Module: FeatureEngineering
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.4
# Last modified : 2021-09-15

def FE0_AutoLags(data = None, ArgsList=None, LagColumnNames = None, DateColumnName = None, ByVariables = None, LagPeriods = 1, ImputeValue = -1, Sort = True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate lags for multiple periods for multiple variables and by variables
    
    # Output
    Return a datatable, polars frame, or pandas frame with new lag columns
    
    # Parameters
    data:           Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:       If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    LagColumnNames: A list of columns that will be lagged
    DateColumnName: Primary date column used for sorting
    ByVariables:    Columns to partition over
    LagPeriods:     List of integers for the lookback lengths
    ImputeValue:    Value to fill the NA's for beginning of series
    Sort:           Sort the Frame before computing the lags - if you're data is sorted set this to False
    Processing:     'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:     'datatable', 'polars', or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    'datatable', 'polars', or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test AutoLags
    import timeit
    import datatable as dt
    from datatable import sort, f, by
    import retrofit
    from retrofit import FeatureEngineering as fe

    ## Group Example:
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoLags(data=data, LagPeriods=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## Group and Multiple Periods and LagColumnNames:
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## No Group Example:
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)
    
    # QA: No Group Case: Step through function
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    LagPeriods = [1, 3, 5]
    LagColumnNames = ['Leads','XREGS1','XREGS2']
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = None
    lp = 1
    ImputeValue = -1
    Sort = True
    
    # QA: Group Case: Step through function
    Processing='datatable'
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
    
    # ArgsList Collection
    if not ArgsList is None:
      LagColumnNames = ArgsList['LagColumnNames']
      DateColumnName = ArgsList['DateColumnName']
      ByVariables = ArgsList['ByVariables']
      LagPeriods = ArgsList['LagPeriods']
      ImputeValue = ArgsList['ImputeValue']
    else:
      ArgsList = dict(
        LagColumnNames = LagColumnNames,
        DateColumnName = DateColumnName,
        ByVariables = ByVariables,
        LagPeriods = LagPeriods,
        ImputeValue = ImputeValue)

    # For making copies of lists so originals aren't modified
    import copy
    
    # Import datatable methods
    if Processing.lower() == 'datatable' or OutputFrame.lower() == 'datatable' or InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import sort, f, by, ifelse

    # Import polars methods
    if Processing.lower() == 'polars' or OutputFrame.lower() == 'polars' or InputFrame.lower() == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)

    # Ensure List
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]

    # Sort data
    if Sort == True and Processing.lower() == 'datatable':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data = data[:, :, sort(SortCols, reverse=rev)]
      else:
        data = data[:, :, sort(DateColumnName, reverse=True)]
    elif Sort == True and Processing.lower() == 'polars':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data.sort(SortCols, reverse = rev, in_place = True)
      else:
        if not isinstance(data[DateColumnName].dtype(), pl.Date32):
          data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
        data.sort(DateColumnName, reverse = True, in_place = True)

    # Ensure List
    if not LagColumnNames is None and not isinstance(LagColumnNames, list):
      LagColumnNames = [LagColumnNames]

    # Ensure List
    if not LagPeriods is None and not isinstance(LagPeriods, list):
      LagPeriods = [LagPeriods]

    # Build lags
    if Processing.lower() == 'datatable':
      for lcn in LagColumnNames:
        for lp in LagPeriods:
          
          # New Column Name
          Ref1 = "Lag_" + str(lp) + "_" + lcn
          
          # Generate lags
          if ByVariables is not None:
            data = data[:, f[:].extend({Ref1: dt.shift(f[lcn], n = lp)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref1: dt.shift(f[lcn], n = lp)})]

          # Impute NA
          if not ImputeValue is None:
            data[Ref1] = data[:, ifelse(f[Ref1] == None, -1, f[Ref1])]

    elif Processing.lower() == 'polars':
      for lcn in LagColumnNames:
        for lp in LagPeriods:
          
          # New Column Name
          Ref1 = "Lag_" + str(lp) + "_" + lcn
          
          # Generate lags
          if ByVariables is not None:
            if not ImputeValue is None:
              data = (data.select([
                pl.all(),
                col(lcn).shift_and_fill(lp, ImputeValue).over(ByVariables).explode().alias(Ref1)]))
            else:
              data = (data.select([
                pl.all(),
                col(lcn).shift(lp).over(ByVariables).explode().alias(Ref1)]))
          else:
            if not ImputeValue is None:
              data = (data.select([
                pl.all(),
                col(lcn).shift_and_fill(lp, ImputeValue).alias(Ref1)]))
            else:
              data = (data.select([
                pl.all(),
                col(lcn).shift(lp).alias(Ref1)]))

    # Convert Frame
    if OutputFrame.lower() == 'pandas' and (Processing.lower() == 'datatable' or Processing.lower() == 'polars'):
      data = data.to_pandas()
    elif OutputFrame.lower() == 'datatable' and Processing.lower() == 'polars':
      data = data.to_pandas()
      data = dt.Frame(data)
    
    # Return data
    return dict(data = data, ArgsList = ArgsList)


def FE0_AutoRollStats(data = None, ArgsList=None, RollColumnNames = None, DateColumnName = None, ByVariables = None, MovingAvg_Periods = None, MovingSD_Periods = None, MovingMin_Periods = None, MovingMax_Periods = None, ImputeValue = -1, Sort = True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
    # Output
    Return a datatable, polars frame, or pandas frame with new rolling statistics columns
    
    # Parameters
    data:             Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:         If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    RollColumnNames:  A list of columns that will be lagged
    DateColumnName:   Primary date column used for sorting
    ByVariables:      Columns to partition over
    Moving_*_Periods: List of integers for look back window
    ImputeValue:      Value to fill the NA's for beginning of series
    Sort:             Sort the Frame before computing the lags - if you're data is sorted set this to False
    Processing:       'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:       'datatable' or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:      'datatable' or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA AutoRollStats
    import timeit
    import datatable as dt
    from datatable import sort, f, by
    import retrofit
    from retrofit import FeatureEngineering as fe

    ## No Group Example
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## Group and Multiple Periods and RollColumnNames:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## No Group Example:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
    t_end = timeit.default_timer()
    t_end - t_start
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

    # ArgsList Collection
    if not ArgsList is None:
      RollColumnName = ArgsList['RollColumnNames']
      DateColumnName = ArgsList['DateColumnName']
      ByVariables = ArgsList['ByVariables']
      MovingAvg_Periods = ArgsList['MovingAvg_Periods']
      MovingSD_Periods = ArgsList['MovingSD_Periods']
      MovingMin_Periods = ArgsList['MovingMin_Periods']
      MovingMax_Periods = ArgsList['MovingMax_Periods']
      ImputeValue = ArgsList['ImputeValue']
    else:
      ArgsList = dict(
        RollColumnNames = RollColumnNames,
        DateColumnName = DateColumnName,
        ByVariables = ByVariables,
        MovingAvg_Periods = MovingAvg_Periods,
        MovingSD_Periods = MovingSD_Periods,
        MovingMin_Periods = MovingMin_Periods,
        MovingMax_Periods = MovingMax_Periods,
        ImputeValue = ImputeValue)

    # For making copies of lists so originals aren't modified
    import copy
    
    # Import datatable methods
    if Processing.lower() == 'datatable' or OutputFrame.lower() == 'datatable' or InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import sort, f, by, ifelse

    # Import polars methods
    if Processing.lower() == 'polars' or OutputFrame.lower() == 'polars' or InputFrame.lower() == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)

    # Ensure List
    if not ByVariables is None and not isinstance(ByVariables, list):
      ByVariables = [ByVariables]

    # Sort data
    if Sort == True and Processing.lower() == 'datatable':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data = data[:, :, sort(SortCols, reverse=rev)]
      else:
        data = data[:, :, sort(DateColumnName, reverse=True)]
    elif Sort == True and Processing.lower() == 'polars':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data.sort(SortCols, reverse = rev, in_place = True)
      else:
        if not isinstance(data[DateColumnName].dtype(), pl.Date32):
          data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
        data.sort(DateColumnName, reverse = True, in_place = True)

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
    MaxVal = max(MovingAvg_Periods, MovingSD_Periods, MovingMin_Periods, MovingMax_Periods)[0]

    # datatable processing
    if Processing.lower() == 'datatable':
      for rcn in RollColumnNames:
        for ns in range(1, MaxVal+1):
          
          # Constants
          Ref = str(ns) + "_" + rcn
          Ref1 = "TEMP__Lag_" + Ref

          # Generate Lags for rowmean, rowsd, rowmin, rowmax
          if ByVariables is not None:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = ns)}), by(ByVariables)]
          else:
            data = data[:, f[:].extend({Ref1: dt.shift(f[rcn], n = ns)})]

          # Rolling Mean
          if ns in MovingAvg_Periods:
            Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
            data = data[:, f[:].extend({"RollMean_" + Ref: dt.rowmean(f[Ref2])})]

          # Rolling SD
          if ns in MovingSD_Periods:
            Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
            data = data[:, f[:].extend({"RollSD_" + Ref: dt.rowsd(f[Ref2])})]

          # Rolling Min
          if ns in MovingMin_Periods:
            Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
            data = data[:, f[:].extend({"RollMin_" + Ref: dt.rowmin(f[Ref2])})]

          # Rolling Max
          if ns in MovingMax_Periods:
            Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
            data = data[:, f[:].extend({"RollMax_" + Ref: dt.rowmax(f[Ref2])})]

        # Remove Temporary Lagged Columns
        del data[:, [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]]

    # polars processing
    elif Processing.lower() == 'polars':
      for rcn in RollColumnNames:
        for ns in range(1, MaxVal+1):
          
          # Constants
          Ref = str(ns) + "_" + rcn
          
          # Rolling Mean
          data = (data.select([
            pl.mean(ns).over(ByVariables).explode().alias(Ref)]))

    # Convert Frame
    if OutputFrame.lower() == 'pandas' and (Processing.lower() == 'datatable' or Processing.lower() == 'polars'):
      data = data.to_pandas()
    elif OutputFrame.lower() == 'datatable' and Processing.lower() == 'polars':
      data = data.to_pandas()
      data = dt.Frame(data)
    
    # Return data
    return dict(data = data, ArgsList = ArgsList)


def FE0_AutoDiff(data = None, ArgsList = None, DateColumnName = None, ByVariables = None, DiffNumericVariables = None, DiffDateVariables = None, DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort = True, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):

    """
    # Goal:
    Automatically generate rolling averages, standard deviations, mins and maxes for multiple periods for multiple variables and by variables
    
    # Output
    Return a datatable, polars frame, or pandas frame with new difference columns
    
    # Parameters
    data:                 Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:             If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    DateColumnName:       Primary date column used for sorting
    ByVariables:          Columns to partition over
    DiffNumericVariables: Numeric variable name scalar or list
    DiffDateVariables:    Date variable name scalar or list
    DiffGroupVariables:   Categorical variable name scalar or list
    NLag1:                Default 0. 0 means the current value - NLag2_Current_Value, otherwise NLag1_Current_Value - NLag2_Current_Value
    NLag2:                Default 1. 1 means a lag1 of the current value
    Sort:                 True or False
    Processing:           'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:           'datatable' or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:          'datatable' or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA AutoDiff
    import timeit
    import datatable as dt
    from datatable import sort, f, by
    import retrofit
    from retrofit import FeatureEngineering as fe

    ## Group Example:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## Group and Multiple Periods and RollColumnNames:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    ## No Group Example:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE0_AutoDiff(data=data, ArgsList=None, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)
    
    # QA: No Group Case: Step through function
    data=data
    ArgsList=None
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
    ArgsList=None
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
    
    # ArgsList Collection
    if not ArgsList is None:
      DateColumnName = ArgsList['DateColumnName']
      ByVariables = ArgsList['ByVariables']
      DiffNumericVariables = ArgsList['DiffNumericVariables']
      DiffDateVariables = ArgsList['DiffDateVariables']
      DiffGroupVariables = ArgsList['DiffGroupVariables']
      NLag1 = ArgsList['NLag1']
      NLag2 = ArgsList['NLag2']
    else:
      ArgsList = dict(
        DateColumnName = DateColumnName,
        ByVariables = ByVariables,
        DiffNumericVariables = DiffNumericVariables,
        DiffDateVariables = DiffDateVariables,
        DiffGroupVariables = DiffGroupVariables,
        NLag1 = NLag1,
        NLag2 = NLag2)

    # For making copies of lists so originals aren't modified
    import copy
    
    # Import datatable methods
    if Processing.lower() == 'datatable' or OutputFrame.lower() == 'datatable' or InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import sort, f, by, ifelse

    # Import polars methods
    if Processing.lower() == 'polars' or OutputFrame.lower() == 'polars' or InputFrame.lower() == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col
    
    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)

    # Ensure List
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

    # Sort data
    if Sort == True and Processing.lower() == 'datatable':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data = data[:, :, sort(SortCols, reverse=rev)]
      else:
        data = data[:, :, sort(DateColumnName, reverse=True)]
    elif Sort == True and Processing.lower() == 'polars':
      if ByVariables is not None:
        SortCols = copy.copy(ByVariables)
        SortCols.extend(DateColumnName)
        rev = [True for t in range(len(SortCols))]
        data.sort(SortCols, reverse = rev, in_place = True)
      else:
        if not isinstance(data[DateColumnName].dtype(), pl.Date32):
          data[DateColumnName] = data[DateColumnName].cast(pl.Date32)
        data.sort(DateColumnName, reverse = True, in_place = True)

    # DiffNumericVariables
    if Processing.lower() == 'datatable':
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
    if OutputFrame.lower() == 'pandas' and (Processing.lower() == 'datatable' or Processing.lower() == 'polars'):
      data = data.to_pandas()
    elif OutputFrame.lower() == 'datatable' and Processing.lower() == 'polars':
      data = data.to_pandas()
      data = dt.Frame(data)
    
    # Return data
    return dict(data = data, ArgsList = ArgsList)


def FE1_AutoCalendarVariables(data = None, ArgsList = None, DateColumnNames = None, CalendarVariables = None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
  
    """
    # Goal:
    Automatically generate calendar variables from your date columns
    
    # Output
    Return a datatable, polars, or pandas frame with new calendar variables
    
    # Parameters
    data:                 Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:             If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    DateColumnNames:      Primary date column used for sorting
    CalendarVariables:    'nanosecond', 'second', 'minute', 'hour', 'mday', 'wday', 'month', 'quarter', 'year'
    Processing:           'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:           'datatable' or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:          'datatable' or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # QA AutoCalendarVariables
    import timeit
    import datatable as dt
    from datatable import sort, f, by, ifelse
    import retrofit
    from retrofit import FeatureEngineering as fe

    ## Example:
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    data = fe.FE1_AutoCalendarVariables(data=data, ArgsList=None, DateColumnNames = 'CalendarDateColumn', CalendarVariables = ['wday','mday','wom','month','quarter','year'], Processing = 'datatable', InputFrame = 'datatable', OutputFrame = 'datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    print(data.names)

    # QA: No Group Case: Step through function
    data=data
    ArgsList=None
    DateColumnNames = 'CalendarDateColumn'
    CalendarVariables = ['wday','mday','wom','month','quarter','year']
    Processing = 'datatable'
    InputFrame = 'datatable'
    OutputFrame = 'datatable'
    DateVar = 'CalendarDateColumn'
    CVars = 'year'
    """
    
    # ArgsList Collection
    if not ArgsList is None:
      DateColumnNames = ArgsList['DateColumnNames']
      CalendarVariables = ArgsList['CalendarVariables']
    else:
      ArgsList = dict(
        DateColumnNames = DateColumnNames,
        CalendarVariables = CalendarVariables)

    # Imports
    import datatable as dt
    from datatable import time, ifelse, f, update
    from datatable import Frame
    
    # Ensure List
    if not DateColumnNames is None and not isinstance(DateColumnNames, list):
      DateColumnNames = [DateColumnNames]

    # Ensure List
    if not CalendarVariables is None and not isinstance(CalendarVariables, list):
      CalendarVariables = [CalendarVariables]

    # Loop through DateColumns
    for DateVar in DateColumnNames:
      for CVars in CalendarVariables:
        
        # Nanosecond
        if(CVars.lower() in 'nanosecond'):
          try:
            data[:, f[:].extend({DateVar + '_nanosecond': time.nanosecond(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.nanosecond calculation due to type mismatch")

        # Second
        if(CVars.lower() in 'second'):
          try:
            data = data[:, f[:].extend({DateVar + '_second': time.second(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.second calculation due to type mismatch")

        # Minute
        if(CVars.lower() in 'minute'):
          try:
            data = data[:, f[:].extend({DateVar + '_minute': time.minute(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.minute calculation due to type mismatch")

        # Hour
        if(CVars.lower() in 'hour'):
          try:
            data = data[:, f[:].extend({DateVar + '_hour': time.hour(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.hour calculation due to type mismatch")

        # day_of_week
        if(CVars.lower() in 'wday'):
          try:
            data = data[:, f[:].extend({DateVar + '_wday': time.day_of_week(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.day_of_week 'wday' calculation due to type mismatch")

        # day of month
        if(CVars.lower() in 'mday'):
          try:
            data = data[:, f[:].extend({DateVar + '_mday': time.day(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.day 'mday' calculation due to type mismatch")

        # month
        if(CVars.lower() in 'month'):
          try:
            data = data[:, f[:].extend({DateVar + '_month': time.month(f[DateVar])})]
          except ValueError:
            raise print("Skipping wday time.month calculation due to type mismatch")

        # quarter
        if(CVars.lower() in 'quarter'):
          try:
            data = data[:, f[:].extend({'temp___temp': time.month(f[DateVar])})]
            data[:, update(temp___temp = ifelse(f['temp___temp'] <= 3, 1, ifelse(f['temp___temp'] <= 6, 2, ifelse(f['temp___temp'] <= 9, 3, 4))))]
            data.names = {'temp___temp': DateVar + '_quarter'}
          except ValueError:
            raise print("Skipping time.month 'quarter' calculation due to type mismatch")

        # year
        if(CVars.lower() in 'year'):
          try:
            data = data[:, f[:].extend({DateVar + '_year': time.year(f[DateVar])})]
          except ValueError:
            raise print("Skipping time.year calculation due to type mismatch")

    # Return
    return dict(data = data, ArgsList = ArgsList)

def FE1_DummyVariables(data=None, ArgsList=None, CategoricalColumnNames=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    """
    # Goal:
    Automatically generate dummy variables for CategoricalColumnNames provided by user
      
    # Output
    Return a datatable
    
    # Parameters
    data:                   Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:               None or Dict. If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    CategoricalColumnNames: Scalar. Primary date column used for sorting
    Processing:             'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:             'datatable', 'polars', or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:            'datatable', 'polars', or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # Example: datatable
    import timeit
    import datatable as dt
    import retrofit
    from retrofit import FeatureEngineering as fe
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    Output = fe.FE1_DummyVariables(data=data, ArgsList=None, CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    data = Output['data']
    ArgsList = Output['ArgsList']
    
    # Example: polars
    import timeit
    import retrofit
    from retrofit import FeatureEngineering as fe
    import polars as pl
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    Output = fe.FE1_DummyVariables(data=data, ArgsList=None, CategoricalColumnNames=['MarketingSegments','MarketingSegments2'], Processing='polars', InputFrame='polars', OutputFrame='polars')
    t_end = timeit.default_timer()
    t_end - t_start
    data = Output['data']
    ArgsList = Output['ArgsList']
    
    # QA
    ArgsList=None
    CategoricalColumnNames=['MarketingSegments','MarketingSegments2']
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    
    Processing='polars'
    InputFrame='polars'
    OutputFrame='polars'
    """
    # ArgsList Collection
    if not ArgsList is None:
      CategoricalColumnNames = ArgsList['CategoricalColumnNames']
    else :
      ArgsList = dict(CategoricalColumnNames=CategoricalColumnNames)
    
    # Import datatable methods
    if Processing.lower() == 'datatable' or OutputFrame.lower() == 'datatable' or InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import split_into_nhot, str

    # Import polars methods
    if Processing.lower() == 'polars' or OutputFrame.lower() == 'polars' or InputFrame.lower() == 'polars':
      import polars as pl
      from polars import col
      from polars.lazy import col

    # Ensure List
    if not CategoricalColumnNames is None and not isinstance(CategoricalColumnNames, list):
      CategoricalColumnNames = [CategoricalColumnNames]

    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)

    # Create dummies
    if Processing.lower() == 'datatable':
      data_new = data.copy()
      for column in CategoricalColumnNames:
        df_ohe = dt.str.split_into_nhot(data_new[column])
        df_ohe.names = [f'{column}_{col}' for col in df_ohe.names]
        data_new.cbind(df_ohe)
    elif Processing.lower() == 'polars':
      for column in CategoricalColumnNames:
        data = data.hstack(pl.get_dummies(data[column]))

    # Convert Frame
    if OutputFrame.lower() == 'pandas' and Processing.lower() == 'datatable': 
      data = data.to_pandas()
    elif OutputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = data.to_pandas()
    elif OutputFrame.lower() == 'datatable' and Processing.lower() == 'polars':
      data = data.to_pandas()
      data = dt.Frame(data)

    # Return data
    if Processing.lower() == 'datatable':
      return dict(data = data_new, ArgsList = ArgsList)
    elif Processing.lower() == 'polars':
      return dict(data = data, ArgsList = ArgsList)

def FE2_AutoDataParition(data=None, ArgsList=None, DateColumnName=None, PartitionType='random', Ratios=None, ByVariables=None, Sort=False, Processing='datatable', InputFrame='datatable', OutputFrame='datatable'):
    
    """
    # Goal:
    Automatically generate train, validation, and test data sets for modeling purposes
      
    # Output
    Return a datatable, polars frame, or pandas frame with new lag columns
    
    # Parameters
    data:           Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    ArgsList:       None or Dict. If running for the first time the function will create an ArgsList dictionary of your specified arguments. If you are running to recreate the same features for model scoring then you can pass in the ArgsList dictionary without specifying the function arguments
    DateColumnName: Scalar. Primary date column used for sorting
    PartitionType:  Scalar. Columns to partition over
    Ratios:         List. Use ths for PartitionType 'random'. List of decimal values for determining how many data goes into each data frame.
    Sort:           Sort data before creating time based partitions
    ByVariables:    None or List. Stratify the data paritioning using ByVariables
    Processing:     'datatable' or 'polars'. Choose the package you want to do your processing
    InputFrame:     'datatable', 'polars', or 'pandas' If you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    'datatable', 'polars', or 'pandas' If you want the output Frame to be pandas change value to 'pandas'
    
    # datatable Example
    import timeit
    import datatable as dt
    import retrofit
    from retrofit import FeatureEngineering as fe
    from retrofit import utils as u
    
    # random
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    DataSets = fe.FE2_AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='random', Ratios=[0.70,0.20,0.10], Sort=False, ByVariables=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    TrainData = DataSets['TrainData']
    ValidationData = DataSets['ValidationData']
    TestData = DataSets['TestData']
    ArgsList = DataSets['ArgsList']
    
    # time
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    DataSets = fe.FE2_AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='time', Ratios=[0.70,0.20,0.10], Sort=True, ByVariables=None, Processing='datatable', InputFrame='datatable', OutputFrame='datatable')
    t_end = timeit.default_timer()
    t_end - t_start
    TrainData = DataSets['TrainData']
    ValidationData = DataSets['ValidationData']
    TestData = DataSets['TestData']
    ArgsList = DataSets['ArgsList']
    
    # polars Example
    import timeit
    import polars as pl
    import retrofit
    from retrofit import FeatureEngineering as fe
    from retrofit import utils as u
    
    # random
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    DataSets = fe.FE2_AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='random', Ratios=[0.70,0.20,0.10], Sort=False, ByVariables=None, Processing='polars', InputFrame='polars', OutputFrame='polars')
    t_end = timeit.default_timer()
    t_end - t_start
    TrainData = DataSets['TrainData']
    ValidationData = DataSets['ValidationData']
    TestData = DataSets['TestData']
    ArgsList = DataSets['ArgsList']
    
    # time
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = pl.read_csv("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    t_start = timeit.default_timer()
    DataSets = fe.FE2_AutoDataParition(data=data, ArgsList=None, DateColumnName='CalendarDateColumn', PartitionType='time', Ratios=[0.70,0.20,0.10], Sort=True, ByVariables=None, Processing='polars', InputFrame='polars', OutputFrame='polars')
    t_end = timeit.default_timer()
    t_end - t_start
    TrainData = DataSets['TrainData']
    ValidationData = DataSets['ValidationData']
    TestData = DataSets['TestData']
    ArgsList = DataSets['ArgsList']
    
    # random
    ArgsList=None
    DateColumnName='CalendarDateColumn'
    PartitionType='random'
    Ratios=[0.70,0.20,0.10]
    ByVariables=None
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    
    Processing='polars'
    InputFrame='polars'
    OutputFrame='polars'
    
    # time
    ArgsList=None
    DateColumnName='CalendarDateColumn'
    PartitionType='time'
    Ratios=[0.70,0.20,0.10]
    ByVariables=None
    Processing='datatable'
    InputFrame='datatable'
    OutputFrame='datatable'
    """
  
    # ArgsList Collection
    if not ArgsList is None:
      DateColumnName = ArgsList['DateColumnName']
      PartitionType = ArgsList['PartitionType']
      Ratios = ArgsList['Ratios']
      ByVariables = ArgsList['ByVariables']
    else :
      ArgsList = dict(
        DateColumnName=DateColumnName,
        PartitionType=PartitionType,
        Ratios=Ratios,
        ByVariables=ByVariables)

    # For making copies of lists so originals aren't modified
    import numpy as np
    from retrofit import utils as u

    # Import datatable methods
    if Processing.lower() == 'datatable' or OutputFrame.lower() == 'datatable' or InputFrame.lower() == 'datatable':
      import datatable as dt
      from datatable import f, by, sort

    # Import polars methods
    if Processing.lower() == 'polars' or OutputFrame.lower() == 'polars' or InputFrame.lower() == 'polars':
      import polars as pl

    # Convert to datatable
    if InputFrame.lower() == 'pandas' and Processing.lower() == 'datatable':
      data = dt.Frame(data)
    elif InputFrame.lower() == 'pandas' and Processing.lower() == 'polars':
      data = pl.from_pandas(data)

    # Accumulate Ratios
    Ratios = u.cumsum(Ratios)

    # datatable
    if Processing.lower() == 'datatable':

      # Random partitioning
      if PartitionType.lower() == 'random':

        # Add random number column
        data = data[:, f[:].extend({"ID": np.random.uniform(0,1, size = data.shape[0])})]

      # Time base partitioning
      if PartitionType.lower() == 'time':

        # Sort data
        if Sort == True:
          data = data[:, :, sort(f[DateColumnName], reverse = False)]

      # Number of rows
      NumRows = data.nrows
          
      # Grab row number boundaries
      TrainRowsMax = NumRows * Ratios[0]
      ValidRowsMax = NumRows * Ratios[1]
        
      # TrainData
      TrainData = data[:int(TrainRowsMax), ...]
      del TrainData[:, 'ID']
        
      # ValidationData
      ValidationData = data[int(TrainRowsMax+1):int(ValidRowsMax), ...]
      del ValidationData[:, 'ID']
        
      # TestData
      if len(Ratios) == 3:
        TestData = data[int(ValidRowsMax):, ...]
        del TestData[:, 'ID']
      else:
        TestData = None

    # polars
    if Processing.lower() == 'polars':
      
      # Random partitioning
      if PartitionType.lower() == 'random':
        
        # Prepare data
        data['ID'] = np.random.uniform(0,1, size = data.shape[0])
        data = data.sort('ID')
        data.drop_in_place('ID')
        
      # Time base partitioning
      if PartitionType.lower() == "time":
        
        # Prepare data
        if Sort == True:
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
    
    # Convert Frame
    if OutputFrame.lower() == 'pandas' and (Processing.lower() == 'datatable' or Processing.lower() == 'polars'):
      TrainData = TrainData.to_pandas()
      ValidationData = ValidationData.to_pandas()
      if len(Ratios) == 3:
        TestData = TestData.to_pandas()
    elif OutputFrame.lower() == 'datatable' and Processing.lower() == 'polars':
      TrainData = TrainData.to_pandas()
      TrainData = dt.Frame(TrainData)
      ValidationData = ValidationData.to_pandas()
      ValidationData = dt.Frame(ValidationData)
      if len(Ratios) == 3:
        TestData = TestData.to_pandas()
        TestData = dt.Frame(TestData)
    
    # Return data
    return dict(TrainData = TrainData, ValidationData = ValidationData, TestData = TestData, ArgsList = ArgsList)
