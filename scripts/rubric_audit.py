"""
rubric_audit.py
────────────────
Audits annotation consistency across all annotators using:
  - Inter-Annotator Agreement (IAA) via Cohen's Kappa
  - Score distribution heatmaps (terminal-friendly)
  - Per-annotator bias detection
  - Gold-standard pair identification (all dims ≥ 4)

Usage:
    python scripts/rubric_audit.py --input data/annotated/export.json
    python scripts/rubric_audit.py --input data/annotated/ --export-gold
"""

import os
import json
import click
import itertools
import statistics
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent

DIMS = ["helpfulness", "accuracy", "coherence", "safety", "conciseness"]
GOLD_THRESHOLD = 4  # All dims must be ≥ this to be gold_standard


# ─────────────────────────────────────────────
# Cohen's Kappa
# ─────────────────────────────────────────────

def cohen_kappa(rater1: list[int], rater2: list[int], n_cats: int = 5) -> float:
    """Compute Cohen's Kappa for two raters with integer ratings 1..n_cats."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    n = len(rater1)
    categories = list(range(1, n_cats + 1))

    # Observed agreement
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement
    pe = sum(
        (rater1.count(c) / n) * (rater2.count(c) / n)
        for c in categories
    )
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def kappa_interpretation(k: float) -> str:
    if k < 0:      return "Poor"
    if k < 0.20:   return "Slight"
    if k < 0.40:   return "Fair"
    if k < 0.60:   return "Moderate"
    if k < 0.80:   return "Substantial"
    return "Almost Perfect"


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_records(input_path: Path) -> list[dict]:
    records = []
    paths = list(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            records.extend(data if isinstance(data, list) else [data])
    return records


def parse_annotations(record: dict) -> list[dict]:
    """Extract all annotations with annotator IDs and rating scores."""
    parsed = []
    for ann in record.get("annotations", []):
        annotator = ann.get("completed_by", {})
        annotator_id = annotator.get("id", "unknown")
        scores = {}
        verdict = "accept"
        tags = []

        for item in ann.get("result", []):
            name = item.get("from_name", "")
            val = item.get("value", {})
            if name in DIMS:
                scores[name] = val.get("rating")
            elif name == "verdict":
                verdict = val.get("choices", ["accept"])[0]
            elif name == "tags":
                tags = val.get("choices", [])

        parsed.append({
            "task_id": record.get("id"),
            "annotator_id": annotator_id,
            "scores": scores,
            "verdict": verdict,
            "tags": tags,
        })
    return parsed


# ─────────────────────────────────────────────
# Audit Functions
# ─────────────────────────────────────────────

def compute_iaa(all_anns: list[dict]) -> dict:
    """
    Compute per-dimension Cohen's Kappa for tasks with 2+ annotations.
    Groups annotations by task, pairs them, and averages kappa.
    """
    task_map: dict[str, list[dict]] = defaultdict(list)
    for ann in all_anns:
        task_map[str(ann["task_id"])].append(ann)

    dim_kappas: dict[str, list[float]] = {d: [] for d in DIMS}

    for task_id, anns in task_map.items():
        if len(anns) < 2:
            continue
        for (a, b) in itertools.combinations(anns, 2):
            for dim in DIMS:
                r1 = a["scores"].get(dim)
                r2 = b["scores"].get(dim)
                if r1 is not None and r2 is not None:
                    k = cohen_kappa([r1], [r2])
                    dim_kappas[dim].append(k)

    return {
        dim: round(statistics.mean(vals), 4) if vals else None
        for dim, vals in dim_kappas.items()
    }


def detect_annotator_bias(all_anns: list[dict]) -> dict:
    """Compute per-annotator mean score for each dimension."""
    annotator_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ann in all_anns:
        aid = str(ann["annotator_id"])
        for dim, score in ann["scores"].items():
            if score is not None:
                annotator_scores[aid][dim].append(float(score))

    result = {}
    for aid, dim_scores in annotator_scores.items():
        result[aid] = {
            dim: round(statistics.mean(vals), 2)
            for dim, vals in dim_scores.items()
        }
    return result


def identify_gold_pairs(records: list[dict], all_anns: list[dict]) -> list[dict]:
    """Identify records where all annotations score ≥ GOLD_THRESHOLD on all dims."""
    task_anns: dict[str, list[dict]] = defaultdict(list)
    for ann in all_anns:
        task_anns[str(ann["task_id"])].append(ann)

    gold_records = []
    for record in records:
        tid = str(record.get("id"))
        anns = task_anns.get(tid, [])
        if not anns:
            continue
        is_gold = all(
            all(
                (ann["scores"].get(dim) or 0) >= GOLD_THRESHOLD
                for dim in DIMS
            )
            for ann in anns
        )
        if is_gold:
            gold_records.append(record)
    return gold_records


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Annotated JSON file or directory")
@click.option("--export-gold", is_flag=True, default=False,
              help="Export gold-standard pairs to a separate file")
@click.option("--gold-output", default=None, help="Path for gold pairs output (JSON)")
def audit(input_path: str, export_gold: bool, gold_output: str | None):
    """Audit annotation consistency and quality across all annotators."""

    path = Path(input_path)
    records = load_records(path)
    logger.info(f"Auditing {len(records)} records...")

    all_anns = []
    for record in records:
        all_anns.extend(parse_annotations(record))

    logger.info(f"Total annotation instances: {len(all_anns)}")

    # ── IAA ──
    iaa = compute_iaa(all_anns)
    iaa_table = Table(title="📐 Inter-Annotator Agreement (Cohen's κ)", style="bold blue", box=box.ROUNDED)
    iaa_table.add_column("Dimension", style="cyan")
    iaa_table.add_column("Kappa (κ)", justify="right")
    iaa_table.add_column("Interpretation", style="yellow")
    for dim, kappa in iaa.items():
        if kappa is not None:
            iaa_table.add_row(dim.capitalize(), f"{kappa:.4f}", kappa_interpretation(kappa))
        else:
            iaa_table.add_row(dim.capitalize(), "N/A", "Insufficient data")
    console.print(iaa_table)

    # ── Annotator Bias ──
    bias = detect_annotator_bias(all_anns)
    if bias:
        bias_table = Table(title="👤 Annotator Score Averages (Bias Detection)", style="bold magenta", box=box.ROUNDED)
        bias_table.add_column("Annotator ID", style="cyan")
        for dim in DIMS:
            bias_table.add_column(dim.capitalize(), justify="right")
        for aid, scores in bias.items():
            bias_table.add_row(aid, *[
                f"{scores.get(dim, 'N/A')}" for dim in DIMS
            ])
        console.print(bias_table)

    # ── Gold Standard ──
    gold = identify_gold_pairs(records, all_anns)
    logger.info(f"Gold-standard pairs identified: {len(gold)} / {len(records)}")
    console.print(f"\n🏅 [bold green]Gold Standard Pairs:[/bold green] {len(gold)} / {len(records)}"
                  f"  ({len(gold)/max(len(records),1)*100:.1f}%)")

    if export_gold and gold:
        out = Path(gold_output) if gold_output else path.parent / "gold_standard.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=2, ensure_ascii=False)
        logger.success(f"Gold pairs saved → {out}")


if __name__ == "__main__":
    audit()
