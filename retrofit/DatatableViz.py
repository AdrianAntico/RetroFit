# Module: DatatableViz
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

import numpy as np
import plotly.express as px
import datatable as dt
from datatable import f, sort, update
import plotly.io as pio
import statsmodels

def ScatterPlot(data=None,
                N=100000,
                XVar=None, 
                YVar=None, 
                FacetColVar=None,
                FacetRowVar=None,
                ColorVar=None,
                SizeVar=None,
                SymbolVar=None,
                HoverStatsVar=None,
                MarginalX=None,
                MarginalY=None,
                TrendLine=None,
                Copula=False):
    """
    # Goal:
    Automatically generate scatterplots from datatable data
    https://plotly.com/python/line-and-scatter/
  
    # Output
    Return a plot object and print to screen
  
    # Parameters
    data:          Source data. Either a datatable frame, polars frame, or pandas frame. The function will run either datatable code or polars code. If your input frame is pandas
    N:             Max number of records to plot
    XVar:          String, column name of the variable to use for the x-axis
    YVar:          String, column name of the variable to use for the y-axis
    FacetColVar:   String, column name of the categorical variables to use to create by-plots across columns as output
    FacetRowVar:   String, column name of the categorical variables to use to create by-plots across rows as output
    ColorVar:      String, column name of the variable to use for the color coding of dots
    SizeVar:       String, column name of the variable to use for the sizing of dots
    HoverStatsVar: String, column name of the variable to use for mouse hovering stats
    MarginalX:     String, 'histogram', 'rug', etc.
    MarginalY:     String, 'histogram', 'rug', etc.
    TrendLine:     String, None or one of 'ols', 'lowess', 'rolling', 'ewm', 'expanding'
    Copula:        Logical, Setting to True will convert your scatterplot data into percent-rank values first and then plot
    """
    
    # Ensure datatable
    if not isinstance(data, dt.Frame):
        raise Exception("data needs to be a datatable frame")

    # Ensure XVar is not None
    if not XVar:
        raise Exception("XVar cannot be None")

    # Ensure XVar is not None
    if not YVar:
        raise Exception("YVar cannot be None")

    # Ensure object is str
    if not isinstance(FacetColVar, (str, type(None))):
        raise Exception("FacetColVar should be a string or None")

    # Ensure object is str
    if not isinstance(FacetRowVar, (str, type(None))):
        raise Exception("FacetRowVar should be a string or None")

    # Ensure object is str
    if not isinstance(ColorVar, (str, type(None))):
        raise Exception("ColorVar should be a string or None")

    # Ensure object is str
    if not isinstance(SizeVar, (str, type(None))):
        raise Exception("SizeVar should be a string or None")

    # Ensure object is str
    if not isinstance(SymbolVar, (str, type(None))):
        raise Exception("SymbolVar should be a string or None")

    # Ensure object is str
    if not isinstance(HoverStatsVar, (str, type(None))):
        raise Exception("HoverStatsVar should be a string or None")

    # Ensure object is str
    if not isinstance(MarginalX, (str, type(None))):
        raise Exception("MarginalX should be a string or None")

    # Ensure object is str
    if not isinstance(MarginalY, (str, type(None))):
        raise Exception("MarginalY should be a string or None")

    # Ensure object is str
    if not isinstance(TrendLine, (str, type(None))):
        raise Exception("TrendLine should be a string or None")
      
    # Ensure logical
    if not isinstance(Copula, bool):
        raise Exception("Copula should be set to True or False")

    # Vars to Keep
    Keep = []
    Keep.append(XVar)
    Keep.append(YVar)
    if ColorVar:
        Keep.append(ColorVar)

    if SizeVar:
        Keep.append(SizeVar)

    if HoverStatsVar:
        Keep.append(HoverStatsVar)

    if FacetColVar:
        Keep.append(FacetColVar)

    if FacetRowVar:
        Keep.append(FacetRowVar)

    # Dedupe Keep list
    Keep = list(set(Keep))
    
    # Shrink data to plot faster
    if N:
        data = data[:, f[:].extend({"ID": np.random.uniform(0, 1, size=data.shape[0])})]
        data = data[: int(N), ...]
        del data[:, "ID"]

    # Copula / Spearman
    if Copula:
        
        # Store row count for 1:n and ni / n
        n = data.shape[0]
        
        # XVar
        data = data[:, f[:], sort(XVar)]
        data = data[:, f[:].extend({f"{XVar}_PercRank": np.arange(1, n+1) / n})]
        Keep.append(f"{XVar}_PercRank")
        XVar = f"{XVar}_PercRank"
        
        # YVar
        data = data[:, f[:], sort(YVar)]
        data = data[:, f[:].extend({f"{YVar}_PercRank": np.arange(1, n+1) / n})]
        Keep.append(f"{YVar}_PercRank")
        YVar = f"{YVar}_PercRank"

    # Convert Keep columns to a pandas frame
    data_pandas = data[:, Keep].to_pandas()
    
    # Build plot object
    fig = px.scatter(data_pandas, x=XVar, y=YVar, color=ColorVar, size=SizeVar, hover_name=HoverStatsVar, facet_col=FacetColVar, facet_row=FacetRowVar, marginal_x=MarginalX, marginal_y=MarginalY, trendline=TrendLine)

    # Generate plot
    fig.show()

    # Return plot object
    return fig
