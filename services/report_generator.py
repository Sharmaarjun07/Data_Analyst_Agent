"""
services/report_generator.py

Generates the deliverables for a completed training run:

    training_report.pdf
    metrics.csv
    feature_importance.csv
    leaderboard.csv        (copied from the global leaderboard, filtered/sorted)
    model_card.md

All files are written into the run's own artifact directory
(models/<run_name>/reports/) so everything for a run stays together.

Usage
-----
    from services.report_generator import generate_full_report

    paths = generate_full_report(
        run_name="RandomForest_a1b2c3d4",
        feature_importance={"age": 0.34, "salary": 0.21},   # optional
        leaderboard=[...],                                   # optional, list of dicts
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from services import model_saver as saver
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


def _reports_dir_for_run(run_name: str) -> Path:
    path = config.MODELS_DIR / run_name / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Individual artifacts
# --------------------------------------------------------------------------- #

def generate_metrics_csv(metrics: dict, run_name: str) -> Path:
    """Write metrics.json's contents out as a flat metrics.csv (metric, value)."""
    path = _reports_dir_for_run(run_name) / "metrics.csv"
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved metrics.csv to %s", path)
    return path


def generate_feature_importance_csv(feature_importance: dict, run_name: str) -> Optional[Path]:
    """Write a ranked feature_importance.csv. Returns None (and logs) if feature_importance is empty."""
    if not feature_importance:
        logger.warning("No feature importance provided for run '%s'; skipping feature_importance.csv.", run_name)
        return None

    path = _reports_dir_for_run(run_name) / "feature_importance.csv"
    df = (
        pd.Series(feature_importance, name="Importance")
        .sort_values(ascending=False, key=abs)
        .rename_axis("Feature")
        .reset_index()
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    df.to_csv(path, index=False)
    logger.info("Saved feature_importance.csv to %s", path)
    return path


def copy_leaderboard_csv(run_name: str, leaderboard: Optional[list] = None) -> Optional[Path]:
    """
    Copy the relevant leaderboard rows into this run's report folder.
    If `leaderboard` isn't provided, falls back to the full global leaderboard
    (models/leaderboard.csv) from model_saver.
    """
    path = _reports_dir_for_run(run_name) / "leaderboard.csv"

    if leaderboard is not None:
        board = pd.DataFrame(leaderboard)
    else:
        board = saver.load_leaderboard()

    if board.empty:
        logger.warning("No leaderboard data available for run '%s'; skipping leaderboard.csv.", run_name)
        return None

    board.to_csv(path, index=False)
    logger.info("Saved leaderboard.csv to %s", path)
    return path


def generate_model_card(
    run_name: str,
    metadata: dict,
    metrics: dict,
    feature_importance: Optional[dict] = None,
) -> Path:
    """Write a human-readable model_card.md summarizing the run."""
    path = _reports_dir_for_run(run_name) / "model_card.md"

    lines = [
        f"# Model Card — {run_name}",
        "",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Overview",
        "",
        f"- **Experiment ID:** {metadata.get('experiment_id', 'N/A')}",
        f"- **Model type:** {metadata.get('model_type', metadata.get('model', 'N/A'))}",
        f"- **Problem type:** {metadata.get('problem_type', 'N/A')}",
        f"- **Target column:** {metadata.get('target', 'N/A')}",
        f"- **Training timestamp:** {metadata.get('training_timestamp', 'N/A')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    if feature_importance:
        lines += ["", "## Top Features", "", "| Rank | Feature | Importance |", "|---|---|---|"]
        ranked = sorted(feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for i, (feature, score) in enumerate(ranked[: config.TOP_FEATURES], start=1):
            lines.append(f"| {i} | {feature} | {score} |")

    model_params = metadata.get("model_params")
    if model_params:
        lines += ["", "## Model Parameters", "", "```", *[f"{k}: {v}" for k, v in model_params.items()], "```"]

    lines += [
        "",
        "## Reproducing Predictions",
        "",
        f"This run's artifacts live under `models/{run_name}/`: `best_model.pkl`, "
        "`preprocessor.pkl`, `feature_names.pkl`, and (if applicable) `label_encoder.pkl`. "
        "Use `services.prediction.predict(run_name, new_df)` to get predictions on new data "
        "transformed the same way as training data.",
    ]

    path.write_text("\n".join(lines))
    logger.info("Saved model_card.md to %s", path)
    return path


def generate_training_report_pdf(
    run_name: str,
    metadata: dict,
    metrics: dict,
    feature_importance: Optional[dict] = None,
    leaderboard: Optional[list] = None,
) -> Optional[Path]:
    """
    Render a one-page(ish) PDF summary: metadata, metrics table, top features,
    and leaderboard. Returns None (and logs) if reportlab isn't installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError:
        logger.error("reportlab is not installed; run `pip install reportlab` to enable PDF reports.")
        return None

    path = _reports_dir_for_run(run_name) / "training_report.pdf"
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Training Report — {run_name}", styles["Title"]),
        Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph("Overview", styles["Heading2"]),
        Paragraph(f"Model type: {metadata.get('model_type', metadata.get('model', 'N/A'))}", styles["Normal"]),
        Paragraph(f"Problem type: {metadata.get('problem_type', 'N/A')}", styles["Normal"]),
        Paragraph(f"Target column: {metadata.get('target', 'N/A')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Metrics", styles["Heading2"]),
    ]

    metrics_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in metrics.items()]
    story.append(_styled_table(metrics_table_data, colors))
    story.append(Spacer(1, 12))

    if feature_importance:
        story.append(Paragraph("Top Features", styles["Heading2"]))
        ranked = sorted(feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_rows = [["Rank", "Feature", "Importance"]] + [
            [str(i), feat, str(score)] for i, (feat, score) in enumerate(ranked[: config.TOP_FEATURES], start=1)
        ]
        story.append(_styled_table(top_rows, colors))
        story.append(Spacer(1, 12))

    if leaderboard:
        story.append(Paragraph("Model Comparison", styles["Heading2"]))
        lb_df = pd.DataFrame(leaderboard)
        lb_rows = [list(lb_df.columns)] + lb_df.astype(str).values.tolist()
        story.append(_styled_table(lb_rows, colors))

    try:
        SimpleDocTemplate(str(path), pagesize=letter).build(story)
    except Exception as exc:
        logger.error("Failed to render training_report.pdf: %s", exc)
        return None

    logger.info("Saved training_report.pdf to %s", path)
    return path


def _styled_table(data: list, colors):
    from reportlab.platypus import Table, TableStyle

    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def generate_full_report(
    run_name: str,
    feature_importance: Optional[dict] = None,
    leaderboard: Optional[list] = None,
) -> dict:
    """
    Generate every report artifact for a completed run in one call:
    training_report.pdf, metrics.csv, feature_importance.csv, leaderboard.csv,
    model_card.md — all written to models/<run_name>/reports/.

    metadata and metrics are loaded from the run's saved artifacts;
    feature_importance and leaderboard are optional inputs (e.g. passed
    straight from the pipeline result) since they aren't persisted by
    model_saver itself.

    Returns a dict of the paths written (values are None for any skipped step).
    """
    run_dir = config.MODELS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No saved run found at {run_dir}")

    metadata = saver.load_metadata(run_dir=run_dir)
    metrics = saver.load_metrics(run_dir=run_dir)

    logger.info("Generating full report for run '%s'...", run_name)

    paths = {
        "metrics_csv": generate_metrics_csv(metrics, run_name),
        "feature_importance_csv": generate_feature_importance_csv(feature_importance or {}, run_name),
        "leaderboard_csv": copy_leaderboard_csv(run_name, leaderboard),
        "model_card": generate_model_card(run_name, metadata, metrics, feature_importance),
        "training_report_pdf": generate_training_report_pdf(
            run_name, metadata, metrics, feature_importance, leaderboard
        ),
    }

    logger.info("Full report generated for run '%s'.", run_name)
    return paths
