"""
setup_label_studio.py
─────────────────────
Initializes a Label Studio project with the annotation interface template,
quality rubric, and project settings from config files.

Usage:
    python scripts/setup_label_studio.py
    python scripts/setup_label_studio.py --project-name "SFT Curation v2"
"""

import os
import sys
import json
import click
import yaml
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from label_studio_sdk import Client

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TEMPLATE_PATH = CONFIG_DIR / "label_studio_template.xml"
SCHEMA_PATH = CONFIG_DIR / "annotation_schema.yaml"


def load_label_config() -> str:
    """Load Label Studio XML interface template."""
    if not TEMPLATE_PATH.exists():
        logger.error(f"Template not found: {TEMPLATE_PATH}")
        sys.exit(1)
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def load_schema() -> dict:
    """Load annotation schema from YAML config."""
    with open(SCHEMA_PATH, "r") as f:
        return yaml.safe_load(f)


def connect_to_label_studio() -> Client:
    """Establish connection to Label Studio instance."""
    host = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")

    if not api_key:
        logger.error("LABEL_STUDIO_API_KEY not set in .env")
        sys.exit(1)

    try:
        client = Client(url=host, api_key=api_key)
        client.check_connection()
        logger.success(f"Connected to Label Studio at {host}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Label Studio: {e}")
        sys.exit(1)


@click.command()
@click.option("--project-name", default="LDIRC – SFT Data Curation", show_default=True,
              help="Name for the Label Studio project")
@click.option("--description", default="Annotation pipeline for SFT instruction-response pairs",
              help="Project description")
@click.option("--force", is_flag=True, default=False,
              help="Delete existing project with same name and recreate")
def setup(project_name: str, description: str, force: bool):
    """Set up Label Studio project for SFT data annotation."""

    logger.info("🚀 Initializing LDIRC Label Studio setup...")

    client = connect_to_label_studio()
    label_config = load_label_config()
    schema = load_schema()

    # Check for existing project
    existing = [p for p in client.get_projects() if p.title == project_name]
    if existing:
        if force:
            logger.warning(f"Deleting existing project: {project_name}")
            existing[0].delete()
        else:
            logger.warning(f"Project '{project_name}' already exists (ID={existing[0].id}). Use --force to recreate.")
            print(f"\n✅ Existing project ID: {existing[0].id}")
            return

    # Create new project
    project = client.start_project(
        title=project_name,
        description=description,
        label_config=label_config,
        expert_instruction=build_rubric_instruction(schema),
        show_instruction=True,
        show_collab_predictions=True,
        maximum_annotations=int(os.getenv("ANNOTATORS_PER_TASK", 2)),
        agreement_threshold=float(os.getenv("INTER_ANNOTATOR_AGREEMENT_THRESHOLD", 0.75)),
    )

    # Save project ID to .env
    env_path = BASE_DIR / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    lines = [l for l in lines if not l.startswith("LABEL_STUDIO_PROJECT_ID=")]
    lines.append(f"LABEL_STUDIO_PROJECT_ID={project.id}")
    env_path.write_text("\n".join(lines) + "\n")

    logger.success(f"✅ Project created: '{project_name}' (ID={project.id})")
    print(f"\n{'='*60}")
    print(f"  Project Name : {project_name}")
    print(f"  Project ID   : {project.id}")
    print(f"  Categories   : {len(schema['categories'])}")
    print(f"  Quality Dims : {len(schema['quality_dimensions'])}")
    print(f"{'='*60}\n")


def build_rubric_instruction(schema: dict) -> str:
    """Build annotator instruction text from schema."""
    dims = schema.get("quality_dimensions", [])
    lines = [
        "## 📋 ANNOTATION RUBRIC\n",
        "Please rate each dimension carefully on a 1–5 scale:\n",
    ]
    for d in dims:
        lines.append(f"**{d['label']}** (weight: {int(d['weight']*100)}%)")
        lines.append(f"  → {d['description']}")
        lines.append(f"  Scale: {d['scale'][0]} (poor) → {d['scale'][-1]} (excellent)\n")

    lines += [
        "## ❌ Rejection Criteria",
        "Reject the response if it contains: PII, harmful content,",
        "incoherent text, off-topic replies, or plagiarized content.\n",
        "## 🏷️ Tagging",
        "- Use **gold_standard** for top-quality pairs (all dims ≥ 4)",
        "- Use **needs_review** when unsure",
        "- Use **flagged** for edge cases requiring discussion",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    setup()
