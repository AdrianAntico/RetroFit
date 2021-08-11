# Module: Cross Row Feature Engineering
# Author: Adrian Antico <adrianantico@gmail.com>
# License: Mozilla Public License 2.0
# Release: RetroFit.FeatureEngineering 0.0.1
# Last modified : 2021-08-11

# Lags on datatable by group variables if requested
# Environment setup
def AutoLags(data = None, LagColumnNames = None, DateColumnName = None, ByVariables = None, N = 1, ImputeValue = -1, Sort = True, InputFrame='datatable', OutputFrame='datatable'):
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
    N:              list of integers for the lookback lengths
    ImputeValue:    value to fill the NA's for beginning of series
    Sort:           sort the Frame before computing the lags - if you're data is sorted set this to False
    IntputFrame:    if you input Frame is 'pandas', it will be converted to a datatable Frame for generating the new columns
    OutputFrame:    if you want the output Frame to be pandas change value to 'pandas'
    
    # QA: Test Function
    import datatable as dt
    from datatable import *
    data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
    
    ## Group Example:
    data = AutoLags(data=data, N=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
    print(data.names)
    
    ## Group and Multiple Periods and LagColumnNames:
    data = AutoLags(data=data, N=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True)
    print(data.names)

    ## No Group Example:
    data = AutoLags(data=data, N=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
    print(data.names)
    
    # QA: No Group Case: Step through function
    N = 1
    LagColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = None
    ns = 1
    ImputeValue = -1
    Sort = True
    
    # QA: Group Case: Step through function
    N = 1
    LagColumnNames = 'Leads'
    scns = 'Leads'
    DateColumnName = 'CalendarDateColumn'
    ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label']
    ns = 1
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
    if Sort == True:
      if ByVariables is not None:
        SortCols = ByVariables
        SortCols.append(DateColumnName)
        data = data[:, :, sort(SortCols, reverse=True)]
      else:
        data = data[:, :, sort(DateColumnName)]
    
    # Prepare column and value references
    if not isinstance(LagColumnNames, list):
      LagColumnNames = [LagColumnNames]
    if not isinstance(N, list):
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

    # Convert Frame
    if OutputFrame == 'pandas': data = data.to_pandas()
    
    # Return data
    return data
