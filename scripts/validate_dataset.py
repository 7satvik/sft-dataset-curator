"""
validate_dataset.py
────────────────────
Validates annotated prompt-response pairs exported from Label Studio.
Applies structural, coherence, quality-score, and safety checks.
Produces a detailed validation report and writes cleaned + rejected subsets.

Usage:
    python scripts/validate_dataset.py --input data/annotated/export.json
    python scripts/validate_dataset.py --input data/annotated/ --strict
"""

import os
import re
import json
import sys
import click
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent

# Thresholds from environment
MIN_PROMPT_LEN = int(os.getenv("MIN_PROMPT_LENGTH", 10))
MAX_PROMPT_LEN = int(os.getenv("MAX_PROMPT_LENGTH", 2048))
MIN_RESPONSE_LEN = int(os.getenv("MIN_RESPONSE_LENGTH", 20))
MAX_RESPONSE_LEN = int(os.getenv("MAX_RESPONSE_LENGTH", 4096))
MIN_QUALITY = float(os.getenv("MIN_QUALITY_SCORE", 0.70))
MIN_COHERENCE = float(os.getenv("MIN_COHERENCE_SCORE", 0.65))

# Weighted quality formula
WEIGHTS = {
    "helpfulness": 0.30,
    "accuracy": 0.25,
    "coherence": 0.20,
    "safety": 0.15,
    "conciseness": 0.10,
}


# ─────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────

class ValidationResult:
    def __init__(self, record_id: str):
        self.record_id = record_id
        self.passed = True
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.quality_score: float = 0.0
        self.coherence_score: float = 0.0

    def fail(self, reason: str):
        self.passed = False
        self.failures.append(reason)

    def warn(self, reason: str):
        self.warnings.append(reason)


def check_length(result: ValidationResult, prompt: str, response: str):
    """Validate prompt and response character lengths."""
    if len(prompt) < MIN_PROMPT_LEN:
        result.fail(f"Prompt too short ({len(prompt)} < {MIN_PROMPT_LEN})")
    if len(prompt) > MAX_PROMPT_LEN:
        result.fail(f"Prompt too long ({len(prompt)} > {MAX_PROMPT_LEN})")
    if len(response) < MIN_RESPONSE_LEN:
        result.fail(f"Response too short ({len(response)} < {MIN_RESPONSE_LEN})")
    if len(response) > MAX_RESPONSE_LEN:
        result.fail(f"Response too long ({len(response)} > {MAX_RESPONSE_LEN})")


def check_quality_score(result: ValidationResult, annotations: dict) -> float:
    """Compute weighted quality score from annotation ratings."""
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, weight in WEIGHTS.items():
        rating = annotations.get(dim)
        if rating is not None:
            weighted_sum += (float(rating) / 5.0) * weight
            total_weight += weight

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    result.quality_score = round(score, 4)
    if score < MIN_QUALITY:
        result.fail(f"Quality score too low ({score:.3f} < {MIN_QUALITY})")
    return score


def check_coherence(result: ValidationResult, prompt: str, response: str) -> float:
    """
    Lightweight coherence check: shared n-gram overlap between prompt and response.
    In production, replace with sentence-transformers cosine similarity.
    """
    def ngrams(text: str, n: int = 2) -> set:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(zip(*[tokens[i:] for i in range(n)]))

    p_grams = ngrams(prompt)
    r_grams = ngrams(response)
    if not p_grams or not r_grams:
        result.warn("Could not compute coherence (empty token set)")
        return 0.5

    overlap = len(p_grams & r_grams) / (len(p_grams | r_grams) + 1e-9)
    # Normalize: responses don't need to repeat prompt verbatim; scale generously
    score = min(1.0, overlap * 8)
    result.coherence_score = round(score, 4)
    if score < MIN_COHERENCE:
        result.warn(f"Low coherence score ({score:.3f} < {MIN_COHERENCE})")
    return score


def check_verdict(result: ValidationResult, verdict: str):
    """Ensure annotator accepted the pair."""
    if verdict == "reject":
        result.fail("Annotator verdict: REJECTED")
    elif verdict == "accept_with_edits":
        result.warn("Annotator verdict: Accept with edits (review recommended)")


def check_safety_flags(result: ValidationResult, rejection_reasons: list):
    """Flag pairs marked with safety-related rejection reasons."""
    dangerous = {"harmful_content", "contains_pii", "plagiarized"}
    flagged = dangerous & set(rejection_reasons or [])
    if flagged:
        result.fail(f"Safety flags present: {', '.join(flagged)}")


def validate_record(record: dict) -> ValidationResult:
    """Run all validation checks on a single annotated record."""
    task_id = str(record.get("id", "unknown"))
    result = ValidationResult(task_id)

    data = record.get("data", {})
    prompt = str(data.get("prompt", "")).strip()
    response = str(data.get("response", "")).strip()

    ann_list = record.get("annotations", [])
    if not ann_list:
        result.fail("No annotations found")
        return result

    # Use latest annotation
    ann = ann_list[-1].get("result", [])
    annotations = {}
    verdict = "accept"
    rejection_reasons = []

    for item in ann:
        name = item.get("from_name", "")
        val = item.get("value", {})
        if name in WEIGHTS:
            annotations[name] = val.get("rating")
        elif name == "verdict":
            verdict = val.get("choices", ["accept"])[0]
        elif name == "rejection_reason":
            rejection_reasons = val.get("choices", [])

    check_length(result, prompt, response)
    check_quality_score(result, annotations)
    check_coherence(result, prompt, response)
    check_verdict(result, verdict)
    check_safety_flags(result, rejection_reasons)

    return result


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def load_records(input_path: Path) -> list[dict]:
    """Load annotated records from JSON or directory of JSONs."""
    records = []
    paths = list(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            records.extend(data if isinstance(data, list) else [data])
    return records


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Path to annotated JSON file or directory")
@click.option("--output-dir", default=None, help="Directory to write cleaned/rejected subsets")
@click.option("--strict", is_flag=True, default=False,
              help="Treat warnings as failures")
@click.option("--report", is_flag=True, default=True,
              help="Print detailed validation report")
def validate(input_path: str, output_dir: str | None, strict: bool, report: bool):
    """Validate annotated prompt-response pairs from Label Studio."""

    path = Path(input_path)
    records = load_records(path)
    logger.info(f"Loaded {len(records)} annotated records for validation.")

    results: list[ValidationResult] = []
    passed_records: list[dict] = []
    failed_records: list[dict] = []

    for record in records:
        res = validate_record(record)
        if strict and res.warnings:
            for w in res.warnings:
                res.fail(f"[strict] {w}")
        results.append(res)
        if res.passed:
            passed_records.append(record)
        else:
            failed_records.append(record)

    # ── Summary stats ──
    total = len(results)
    n_passed = len(passed_records)
    n_failed = len(failed_records)
    avg_quality = sum(r.quality_score for r in results) / max(total, 1)
    avg_coherence = sum(r.coherence_score for r in results) / max(total, 1)

    if report:
        console.print(Panel.fit(
            f"[bold cyan]📊 LDIRC Validation Report[/bold cyan]\n\n"
            f"  Total Records   : [white]{total}[/white]\n"
            f"  ✅ Passed       : [green]{n_passed} ({n_passed/total*100:.1f}%)[/green]\n"
            f"  ❌ Failed       : [red]{n_failed} ({n_failed/total*100:.1f}%)[/red]\n"
            f"  Avg Quality     : [yellow]{avg_quality:.3f}[/yellow]\n"
            f"  Avg Coherence   : [yellow]{avg_coherence:.3f}[/yellow]",
            box=box.ROUNDED,
        ))

        # Top failures
        failure_counts: dict[str, int] = {}
        for r in results:
            for f in r.failures:
                key = f.split("(")[0].strip()
                failure_counts[key] = failure_counts.get(key, 0) + 1

        if failure_counts:
            table = Table(title="❌ Failure Breakdown", style="red")
            table.add_column("Reason", style="white")
            table.add_column("Count", justify="right")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                table.add_row(reason, str(count))
            console.print(table)

    # ── Write outputs ──
    out_dir = Path(output_dir) if output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if passed_records:
        out_clean = out_dir / "validated_clean.json"
        with open(out_clean, "w", encoding="utf-8") as f:
            json.dump(passed_records, f, indent=2, ensure_ascii=False)
        logger.success(f"✅ Clean records saved → {out_clean}")

    if failed_records:
        out_fail = out_dir / "validated_rejected.json"
        with open(out_fail, "w", encoding="utf-8") as f:
            json.dump(failed_records, f, indent=2, ensure_ascii=False)
        logger.warning(f"⚠️  Rejected records saved → {out_fail}")


if __name__ == "__main__":
    validate()
