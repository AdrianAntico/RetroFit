# Module: Cross Row Feature Engineering
# Author: Adrian Antico <adrianantico@gmail.com>
# License: Mozilla Public License 2.0
# Release: RetroFit.FeatureEngineering 0.0.1
# Last modified : 2021-08-11

# Lags on datatable by group variables if requested
def AutoLags(data = None, LagColumnNames = None, DateColumnName = None, ByVariables = None, N = 1, ImputeValue = -1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
    """
    # Output
    Return data with additional lag columns, by ByVariables
    
    # Parameters
    data:           is your source datatable
    LagColumnNames: a list of columns that will be lagged
    DateColumnName: primary date column used for sorting
    ByVariables:    columns to lag by
    N:              list of integers for the lookback lengths
    ImputeValue:    value to fill the NA's for beginning of series
    Sort:           sort the Frame before computing the lags - if you're data is sorted set this to False
    IntputFrame:    if you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    if you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test Function
    import datatable as dt
    from datatable import *
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    data = AutoLags(data=data, N=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
    
    # QA: Step through function
    N = 1
    LagColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    ns = 1
    ImputeValue = -1
    Sort = True
    
    """
    
    # Environment setup
    import datatable as dt
    from datatable import *
    
    # Convert to datatable
    if InputFrame == 'pandas': 
      data = dt.Frame(data)

    # Sort data if requested
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName)]
    
    # Prepare column and value references
    LagColumnNames = [LagColumnNames]
    N = [N]
    Cols = data.names
    
    # Build lags
    for lcn in LagColumnNames:
      colnum = Cols.index(lcn)
      for ns in N:
        if ByVariables is not None:
          data = data[:, f[:].extend({"Lag_" + str(ns) + "_" + lcn: dt.shift(f[colnum], n = ns)}), by(ByVariables)]
        else:
          data = data[:, f[:].extend({"Lag_" + str(ns) + "_" + lcn: dt.shift(f[colnum], n = ns)})]

    # Done
    if OutputFrame == 'pandas': data = data.to_pandas()
    return data
