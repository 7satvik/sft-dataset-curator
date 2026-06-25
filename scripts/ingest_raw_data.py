"""
ingest_raw_data.py
──────────────────
Reads raw unstructured text files (JSON, JSONL, TXT, CSV),
normalizes them into prompt-response pairs, and uploads them
as annotation tasks to Label Studio.

Usage:
    python scripts/ingest_raw_data.py --input data/raw/
    python scripts/ingest_raw_data.py --input data/raw/ --batch-size 20 --dry-run
"""

import os
import csv
import json
import sys
import re
import click
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Iterator
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track
from label_studio_sdk import Client

load_dotenv()
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")


# ─────────────────────────────────────────────
# Parsers for each raw format
# ─────────────────────────────────────────────

def parse_jsonl(path: Path) -> Iterator[dict]:
    """Parse .jsonl files with prompt/response fields."""
    with jsonlines.open(path) as reader:
        for i, obj in enumerate(reader):
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input", "")
            response = obj.get("response") or obj.get("output") or obj.get("completion", "")
            if prompt and response:
                yield normalize_pair(prompt, response, source=path.name, idx=i)


def parse_json(path: Path) -> Iterator[dict]:
    """Parse .json files (list or dict formats)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    for i, obj in enumerate(items):
        prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question", "")
        response = obj.get("response") or obj.get("answer") or obj.get("output", "")
        if prompt and response:
            yield normalize_pair(prompt, response, source=path.name, idx=i)


def parse_csv(path: Path) -> Iterator[dict]:
    """Parse CSV with 'prompt' and 'response' columns."""
    df = pd.read_csv(path)
    # Flexible column matching
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("prompt", "instruction", "input", "question"):
            col_map["prompt"] = col
        elif cl in ("response", "output", "completion", "answer"):
            col_map["response"] = col

    if "prompt" not in col_map or "response" not in col_map:
        logger.warning(f"CSV {path.name} missing required columns. Skipping.")
        return

    for i, row in df.iterrows():
        prompt = str(row[col_map["prompt"]]).strip()
        response = str(row[col_map["response"]]).strip()
        if prompt and response:
            yield normalize_pair(prompt, response, source=path.name, idx=i)


def parse_txt(path: Path) -> Iterator[dict]:
    """
    Parse plain text files. Expects blocks separated by '---' or '==='.
    Each block: first line is prompt, rest is response.
    """
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?:---+|===+)\n", content)
    for i, block in enumerate(blocks):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        prompt = lines[0].strip()
        response = "\n".join(lines[1:]).strip()
        if prompt and response:
            yield normalize_pair(prompt, response, source=path.name, idx=i)


PARSERS = {
    ".jsonl": parse_jsonl,
    ".json": parse_json,
    ".csv": parse_csv,
    ".txt": parse_txt,
}


def normalize_pair(prompt: str, response: str, source: str, idx: int) -> dict:
    """Create a standardized task dict for Label Studio upload."""
    return {
        "data": {
            "prompt": prompt.strip(),
            "response": response.strip(),
            "meta": {
                "source_file": source,
                "source_index": idx,
                "char_count_prompt": len(prompt),
                "char_count_response": len(response),
            },
        }
    }


def collect_raw_files(input_dir: Path) -> list[Path]:
    """Recursively collect all supported raw files."""
    files = []
    for ext in PARSERS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def upload_tasks(client: Client, project_id: int, tasks: list[dict]) -> int:
    """Upload tasks to Label Studio in a single batch."""
    project = client.get_project(project_id)
    project.import_tasks(tasks)
    return len(tasks)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option("--input", "input_dir", default=str(RAW_DATA_DIR), show_default=True,
              type=click.Path(exists=True), help="Directory containing raw data files")
@click.option("--batch-size", default=50, show_default=True, help="Upload batch size")
@click.option("--dry-run", is_flag=True, default=False,
              help="Parse files but don't upload to Label Studio")
@click.option("--limit", default=None, type=int,
              help="Max number of tasks to process (useful for testing)")
def ingest(input_dir: str, batch_size: int, dry_run: bool, limit: int | None):
    """Parse raw data files and upload tasks to Label Studio."""

    input_path = Path(input_dir)
    raw_files = collect_raw_files(input_path)

    if not raw_files:
        logger.warning(f"No supported files found in {input_path}")
        sys.exit(0)

    logger.info(f"Found {len(raw_files)} raw file(s) in {input_path}")

    # Parse all files
    all_tasks: list[dict] = []
    stats = {}

    for path in track(raw_files, description="Parsing files..."):
        parser = PARSERS[path.suffix.lower()]
        file_tasks = list(parser(path))
        stats[path.name] = len(file_tasks)
        all_tasks.extend(file_tasks)

    if limit:
        all_tasks = all_tasks[:limit]

    # Print summary table
    table = Table(title="📂 Raw Data Ingestion Summary", style="bold cyan")
    table.add_column("File", style="white")
    table.add_column("Tasks Parsed", justify="right", style="green")
    for fname, count in stats.items():
        table.add_row(fname, str(count))
    table.add_row("─" * 30, "─" * 12)
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(all_tasks)}[/bold]")
    console.print(table)

    if dry_run:
        logger.info("✅ Dry run complete. No tasks uploaded.")
        return

    # Connect and upload
    client = Client(
        url=os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080"),
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
    )
    project_id = int(os.getenv("LABEL_STUDIO_PROJECT_ID", 1))

    # Upload in batches
    total_uploaded = 0
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i: i + batch_size]
        n = upload_tasks(client, project_id, batch)
        total_uploaded += n
        logger.info(f"Uploaded batch {i // batch_size + 1}: {n} tasks")

    logger.success(f"✅ Ingestion complete. {total_uploaded} tasks uploaded to project {project_id}.")


if __name__ == "__main__":
    ingest()
