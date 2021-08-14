# QA: Test AutoLags
import datatable as dt
from datatable import sort, f

## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoLags(data=data, LagPeriods=[1,3,5,7], LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
print(data.names)

## Group and Multiple Periods and LagColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoLags(data=data, LagPeriods=[1,3,5], LagColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoLags(data=data, LagPeriods=1, LagColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, ImputeValue=-1, Sort=True)
print(data.names)


# QA AutoRollStats
import datatable as dt
from datatable import sort, f, by

## No Group Example
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)

## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoRollStats(data=data, RollColumnNames=['Leads','XREGS1'], DateColumnName='CalendarDateColumn', ByVariables=['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoRollStats(data=data, RollColumnNames='Leads', DateColumnName='CalendarDateColumn', ByVariables=None, MovingAvg_Periods=[3,5,7], MovingSD_Periods=[3,5,7], MovingMin_Periods=[3,5,7], MovingMax_Periods=[3,5,7], ImputeValue=-1, Sort=True)
print(data.names)


# QA AutoDiff
import datatable as dt
from datatable import sort, f, by
    
## Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)

## Group and Multiple Periods and RollColumnNames:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = ['MarketingSegments', 'MarketingSegments2', 'MarketingSegments3', 'Label'], DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)

## No Group Example:
data = dt.fread("C:/Users/Bizon/Documents/GitHub/BenchmarkData.csv")
data = AutoDiff(data=data, DateColumnName = 'CalendarDateColumn', ByVariables = None, DiffNumericVariables = 'Leads', DiffDateVariables = 'CalendarDateColumn', DiffGroupVariables = None, NLag1 = 0, NLag2 = 1, Sort=True, InputFrame = 'datatable', OutputFrame = 'datatable')
print(data.names)




