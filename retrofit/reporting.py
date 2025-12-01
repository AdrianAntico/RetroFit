from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import polars as pl
import importlib.resources as pkg_resources
from pathlib import Path


@dataclass
class TableSpec:
    """
    Generic table: list of columns + list of rows (plain Python dicts).
    """
    columns: List[str]
    rows: List[Dict[str, Any]]


@dataclass
class PlotSpec:
    """
    A single ECharts plot for the report.
    """
    title: str
    description: Optional[str]
    echarts_option: Dict[str, Any]


@dataclass
class MetricsSection:
    """
    Wrapper so we can later hang notes / warnings next to a metrics table.
    """
    table: TableSpec
    notes: Optional[str] = None


@dataclass
class ModelInsightsBundle:
    """
    For now: regression-only bundle.
    You can extend this later for binary / multiclass as needed.
    """
    problem_type: str           # "regression" for now
    model_name: str
    model_type: str
    target_col: str
    run_id: Optional[str]
    args: Dict[str, Any]

    # Data
    data_summary: TableSpec
    feature_summary: TableSpec

    # Performance
    metrics: MetricsSection
    calibration_table: TableSpec
    calibration_plot: Optional[PlotSpec]
    residuals_plot: Optional[PlotSpec]
    actual_vs_pred_plot: Optional[PlotSpec]
    residual_dist_plot: Optional[PlotSpec]

    # Feature insights
    feature_importance_table: TableSpec
    pdp_numeric_plots: List[PlotSpec]
    pdp_categorical_plots: List[PlotSpec]

    # Extra hook if you want it later
    extra: Dict[str, Any]


def df_to_table(df: pl.DataFrame) -> TableSpec:
    """
    Convert a Polars DataFrame into TableSpec.
    """
    if df.is_empty():
        return TableSpec(columns=[], rows=[])
    cols = df.columns
    rows = [dict(zip(cols, row)) for row in df.rows()]
    return TableSpec(columns=cols, rows=rows)

