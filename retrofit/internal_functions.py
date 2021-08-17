# Inner function for AutoRollStats
def _RollStatSingleInstance(data, rcn, ns, ByVariables, ColsOriginal, MovingAvg_Periods_, MovingSD_Periods_, MovingMin_Periods_, MovingMax_Periods_):

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
    Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
    data = data[:, f[:].extend({"RollMean_" + Ref: dt.rowmean(f[Ref2])})]
    
  # Rolling SD
  if ns in MovingSD_Periods_:
    Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
    data = data[:, f[:].extend({"RollSD_" + Ref: dt.rowsd(f[Ref2])})]
    
  # Rolling Min
  if ns in MovingMin_Periods_:
    Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
    data = data[:, f[:].extend({"RollMin_" + Ref: dt.rowmin(f[Ref2])})]
    
  # Rolling Max
  if ns in MovingMax_Periods_:
    Ref2 = [zzz for zzz in data.names if 'TEMP__Lag_' in zzz]
    data = data[:, f[:].extend({"RollMax_" + Ref: dt.rowmax(f[Ref2])})]
    
  # Return
  return data
