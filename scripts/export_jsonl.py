"""
export_jsonl.py
────────────────
Exports validated, annotated pairs from Label Studio into model-ready
JSONL datasets in SFT (Supervised Fine-Tuning) format.

Supported output schemas:
  - alpaca    : { "instruction": ..., "input": ..., "output": ... }
  - sharegpt  : { "conversations": [{"from": "human", ...}, {"from": "gpt", ...}] }
  - openai    : { "messages": [{"role": "user", ...}, {"role": "assistant", ...}] }
  - raw       : { "prompt": ..., "response": ..., "meta": {...} }

Usage:
    python scripts/export_jsonl.py
    python scripts/export_jsonl.py --schema openai --split --output data/exports/
    python scripts/export_jsonl.py --schema alpaca --filter-tag gold_standard
"""

import os
import json
import math
import random
import click
import jsonlines
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from label_studio_sdk import Client

load_dotenv()
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_DIR / os.getenv("EXPORT_DIR", "data/exports")

TRAIN_RATIO = float(os.getenv("SPLIT_RATIO_TRAIN", 0.85))
VAL_RATIO = float(os.getenv("SPLIT_RATIO_VAL", 0.10))
TEST_RATIO = float(os.getenv("SPLIT_RATIO_TEST", 0.05))
VERSION = os.getenv("DATASET_VERSION", "v1.0")


# ─────────────────────────────────────────────
# Schema Converters
# ─────────────────────────────────────────────

def to_alpaca(prompt: str, response: str, meta: dict) -> dict:
    """Stanford Alpaca format."""
    return {
        "instruction": prompt,
        "input": "",
        "output": response,
        "meta": meta,
    }


def to_sharegpt(prompt: str, response: str, meta: dict) -> dict:
    """ShareGPT conversation format."""
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ],
        "meta": meta,
    }


def to_openai(prompt: str, response: str, meta: dict) -> dict:
    """OpenAI chat fine-tuning format."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "meta": meta,
    }


def to_raw(prompt: str, response: str, meta: dict) -> dict:
    """Raw key-value format."""
    return {
        "prompt": prompt,
        "response": response,
        "meta": meta,
    }


SCHEMAS = {
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "openai": to_openai,
    "raw": to_raw,
}


# ─────────────────────────────────────────────
# Label Studio Export
# ─────────────────────────────────────────────

def fetch_from_label_studio(filter_tag: str | None = None) -> list[dict]:
    """Pull validated tasks from Label Studio."""
    client = Client(
        url=os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080"),
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
    )
    project_id = int(os.getenv("LABEL_STUDIO_PROJECT_ID", 1))
    project = client.get_project(project_id)

    tasks = project.get_tasks()
    logger.info(f"Fetched {len(tasks)} tasks from Label Studio project {project_id}")

    if filter_tag:
        tasks = [
            t for t in tasks
            if filter_tag in _get_tags(t)
        ]
        logger.info(f"After tag filter '{filter_tag}': {len(tasks)} tasks")

    return tasks


def _get_tags(task: dict) -> list[str]:
    """Extract tags from the latest annotation."""
    for ann in task.get("annotations", []):
        for item in ann.get("result", []):
            if item.get("from_name") == "tags":
                return item.get("value", {}).get("choices", [])
    return []


def _extract_pair(task: dict) -> tuple[str, str, dict]:
    """Extract prompt, response, and metadata from a task."""
    data = task.get("data", {})
    prompt = str(data.get("prompt", "")).strip()
    response = str(data.get("response", "")).strip()

    ann_list = task.get("annotations", [])
    ann_meta = {}
    if ann_list:
        for item in ann_list[-1].get("result", []):
            name = item.get("from_name", "")
            val = item.get("value", {})
            if name in ("helpfulness", "accuracy", "coherence", "safety", "conciseness"):
                ann_meta[name] = val.get("rating")
            elif name == "domain_category":
                ann_meta["domain"] = val.get("choices", [None])[0]
            elif name == "verdict":
                ann_meta["verdict"] = val.get("choices", ["accept"])[0]
            elif name == "tags":
                ann_meta["tags"] = val.get("choices", [])

    meta = {
        "task_id": task.get("id"),
        "source": data.get("meta", {}).get("source_file"),
        "annotations": ann_meta,
        "export_version": VERSION,
        "exported_at": datetime.utcnow().isoformat() + "Z",
    }
    return prompt, response, meta


# ─────────────────────────────────────────────
# JSONL Writer
# ─────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(records)
    logger.success(f"Wrote {len(records)} records → {path}")


def split_dataset(records: list[dict]) -> tuple[list, list, list]:
    """Shuffle and split into train/val/test subsets."""
    random.seed(42)
    random.shuffle(records)
    n = len(records)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val = math.floor(n * VAL_RATIO)
    train = records[:n_train]
    val = records[n_train: n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option("--schema", default="openai", show_default=True,
              type=click.Choice(list(SCHEMAS.keys())),
              help="Output JSONL schema format")
@click.option("--output", "output_dir", default=str(EXPORT_DIR), show_default=True,
              help="Directory to write exported JSONL files")
@click.option("--split", is_flag=True, default=False,
              help="Split output into train/val/test subsets")
@click.option("--filter-tag", default=None,
              help="Only export tasks with this tag (e.g. 'gold_standard')")
@click.option("--from-file", "from_file", default=None, type=click.Path(exists=True),
              help="Load from local validated_clean.json instead of Label Studio")
def export(schema: str, output_dir: str, split: bool, filter_tag: str | None, from_file: str | None):
    """Export validated SFT data to model-ready JSONL format."""

    converter = SCHEMAS[schema]
    out_path = Path(output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Load tasks
    if from_file:
        with open(from_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        logger.info(f"Loaded {len(tasks)} records from {from_file}")
    else:
        tasks = fetch_from_label_studio(filter_tag=filter_tag)

    if not tasks:
        logger.warning("No tasks to export. Exiting.")
        return

    # Convert to target schema
    records = []
    skipped = 0
    for task in tasks:
        try:
            prompt, response, meta = _extract_pair(task)
            if not prompt or not response:
                skipped += 1
                continue
            records.append(converter(prompt, response, meta))
        except Exception as e:
            logger.warning(f"Skipping task {task.get('id')}: {e}")
            skipped += 1

    logger.info(f"Converted {len(records)} records ({skipped} skipped)")

    # Write output
    if split:
        train, val, test = split_dataset(records)
        write_jsonl(train, out_path / f"train_{schema}_{VERSION}_{timestamp}.jsonl")
        write_jsonl(val,   out_path / f"val_{schema}_{VERSION}_{timestamp}.jsonl")
        write_jsonl(test,  out_path / f"test_{schema}_{VERSION}_{timestamp}.jsonl")

        table = Table(title="📦 Dataset Split Summary", style="bold green")
        table.add_column("Split", style="cyan")
        table.add_column("Records", justify="right")
        table.add_column("% of Total", justify="right")
        for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
            table.add_row(name, str(len(subset)), f"{len(subset)/len(records)*100:.1f}%")
        console.print(table)
    else:
        out_file = out_path / f"dataset_{schema}_{VERSION}_{timestamp}.jsonl"
        write_jsonl(records, out_file)

    logger.success(f"✅ Export complete. Schema: [{schema}] | Records: {len(records)}")


if __name__ == "__main__":
    export()
