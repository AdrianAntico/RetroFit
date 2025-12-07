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

    (Currently not used directly by the bundle fields, which store HTML
    strings, but kept here in case we later want to pass raw option dicts.)
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
    Unified bundle for regression and classification model insights.

    problem_type is typically "regression" or "classification".
    """
    problem_type: str
    model_name: str
    model_type: str
    target_col: str
    run_id: Optional[str]
    args: Dict[str, Any]

    # Data
    data_summary: TableSpec
    feature_summary: TableSpec

    # Performance (shared)
    metrics: MetricsSection
    calibration_table: TableSpec

    # Plots are stored as rendered HTML snippets
    calibration_plot: Optional[str]
    residuals_plot: Optional[str]
    actual_vs_pred_plot: Optional[str]
    residual_dist_plot: Optional[str]
    prediction_dist_plot: Optional[str]

    # Classification-specific performance
    roc_plot: Optional[str]
    pr_plot: Optional[str]
    threshold_metrics_plot: Optional[str]
    
    # Feature insights
    feature_importance_table: TableSpec
    interaction_importance_table: Optional[TableSpec]
    pdp_numeric_plots: List[Dict[str, Any]]
    pdp_categorical_plots: List[Dict[str, Any]]

    # SHAP / interpretability
    shap_summary_table: Optional[TableSpec]
    shap_summary_plot: Optional[str]
    shap_dependence_plots: Optional[List[Dict[str, Any]]]

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
