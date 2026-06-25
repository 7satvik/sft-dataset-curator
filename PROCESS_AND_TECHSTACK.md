# 🧠 LOCAL DATA INSTRUCTION RESPONSE CURATION (LDIRC) Tool
### Process Documentation & Technical Stack Reference

> **Built as a production-grade data annotation pipeline** for converting raw unstructured text into structured Supervised Fine-Tuning (SFT) datasets — powering better local language model alignment.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Full Process Walkthrough](#3-full-process-walkthrough)
4. [Tech Stack](#4-tech-stack)
5. [Annotation Rubric](#5-annotation-rubric)
6. [Output Schema Reference](#6-output-schema-reference)
7. [Project File Structure](#7-project-file-structure)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Metrics & Quality Gates](#9-metrics--quality-gates)
10. [Quickstart Commands](#10-quickstart-commands)

---

## 1. Project Overview

The **LDIRC Tool** is an end-to-end annotation pipeline designed to:

| Goal | Details |
|------|---------|
| **Ingest** | Parse raw text from JSON, JSONL, CSV, and TXT sources |
| **Annotate** | Use Label Studio with a custom 5-dimension quality rubric |
| **Validate** | Apply structural, coherence, and quality-score filters |
| **Audit** | Measure inter-annotator agreement (Cohen's Kappa) and detect bias |
| **Export** | Generate model-ready JSONL in Alpaca, ShareGPT, or OpenAI format |

The pipeline was built to curate **100+ prompt-response pairs** across diverse domains including:
`factual_qa` · `code_generation` · `reasoning` · `summarization` · `creative_writing` · `math_problem_solving` · `instruction_following` · `conversation`

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                          │
│   .json  ·  .jsonl  ·  .csv  ·  .txt  (prompt + response)       │
└────────────────────────────┬─────────────────────────────────────┘
                             │  ingest_raw_data.py
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LABEL STUDIO (localhost:8080)                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Custom XML Interface (label_studio_template.xml)       │     │
│  │  ┌───────────┐  ┌──────────────────────────────────┐    │     │
│  │  │  Prompt   │  │  5-Dimension Rating Rubric       │    │     │
│  │  │  Display  │  │  Helpfulness · Accuracy          │    │     │
│  │  │           │  │  Coherence · Safety · Conciseness│    │     │
│  │  └───────────┘  └──────────────────────────────────┘    │     │
│  └─────────────────────────────────────────────────────────┘     │
└────────────────────────────┬─────────────────────────────────────┘
                             │  Annotators complete tasks
                             ▼
              ┌──────────────────────────┐
              │   validate_dataset.py    │
              │  ─ Length checks         │
              │  ─ Quality score gate    │
              │  ─ Coherence scoring     │
              │  ─ Safety flag filter    │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │    rubric_audit.py       │
              │  ─ Cohen's Kappa (IAA)   │
              │  ─ Annotator bias detect │
              │  ─ Gold pair extraction  │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │    export_jsonl.py       │
              │  ─ Alpaca format         │
              │  ─ ShareGPT format       │
              │  ─ OpenAI format         │
              │  ─ Train/Val/Test split  │
              └──────────────────────────┘
                           │
                    model-ready .jsonl
```

---

## 3. Full Process Walkthrough

### Phase 1 — Raw Data Ingestion · `ingest_raw_data.py`

Raw unstructured text arrives in various formats. The ingestion layer normalizes them into a canonical `{ prompt, response }` structure:

| Input Format | Parser Strategy |
|---|---|
| `.jsonl` | Reads line-by-line; maps `instruction/input/prompt` → prompt field |
| `.json` | Handles both list and dict structures |
| `.csv` | Flexible column matching (`instruction`, `question`, `prompt`, etc.) |
| `.txt` | Block-split on `---` or `===` delimiters; first line = prompt |

Each parsed pair is wrapped with provenance metadata (source file, index, character counts) and uploaded to Label Studio in configurable batches.

---

### Phase 2 — Label Studio Setup · `setup_label_studio.py`

The project is initialized programmatically via the Label Studio SDK:

- **Interface:** Custom XML template with styled prompt/response panels, star rating scales for each quality dimension, choice selectors for domain, verdict, and tags.
- **Project config:** Sets `maximum_annotations=2` (two annotators per task) and `agreement_threshold=0.75`.
- **Instruction panel:** Auto-generated from `annotation_schema.yaml`.

---

### Phase 3 — Annotation with Rubric

Human annotators evaluate each pair on **5 weighted quality dimensions**:

| Dimension | Weight | What to Assess |
|-----------|--------|---------------|
| **Helpfulness** | 30% | Does the response fully address the user's intent? |
| **Factual Accuracy** | 25% | Is the information correct and grounded in reality? |
| **Coherence & Fluency** | 20% | Is it logically structured and grammatically sound? |
| **Safety & Harmlessness** | 15% | Free from harmful, biased, or toxic content? |
| **Conciseness** | 10% | Appropriately brief without losing information? |

Each dimension is rated **1–5**. Annotators also select a Domain Category, provide a Verdict (Accept / Accept with Edits / Reject), apply Tags, and add free-text notes.

---

### Phase 4 — Validation & Coherence Scoring · `validate_dataset.py`

Every record is passed through a multi-layer validation pipeline:

```
Record
  ├─ Length Check        → prompt: 10–2048 chars, response: 20–4096 chars
  ├─ Quality Score Gate  → weighted dim average ≥ 0.70
  ├─ Coherence Score     → bigram overlap ≥ 0.65
  ├─ Verdict Check       → "reject" = hard fail
  └─ Safety Flag Check   → "harmful_content", "contains_pii" = hard fail
```

**Outputs:** `validated_clean.json` (passed) · `validated_rejected.json` (failed)

---

### Phase 5 — Rubric Audit & IAA · `rubric_audit.py`

Quality of the annotation process itself is measured:

- **Cohen's Kappa (κ)** per dimension — target ≥ 0.60 (Substantial agreement)
- **Annotator bias detection** — per-annotator mean score analysis
- **Gold pair extraction** — all dims ≥ 4 on all annotations → `gold_standard.json`

---

### Phase 6 — JSONL Export · `export_jsonl.py`

Validated records are exported in 3 SFT-compatible formats with 85/10/5 train/val/test splits. Output files are versioned with timestamps.

---

## 4. Tech Stack

### Core Infrastructure

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Annotation Platform** | Label Studio 1.11 | Web-based annotation UI, task management |
| **SDK** | `label-studio-sdk 0.0.32` | Programmatic project creation & export |
| **Runtime** | Python 3.11+ | All pipeline scripts |

### Data Processing

| Library | Version | Role |
|---------|---------|------|
| `pandas` | 2.2.2 | CSV parsing, tabular data manipulation |
| `jsonlines` | 4.0.0 | Streaming JSONL read/write |
| `pydantic` | 2.7.1 | Data validation and schema enforcement |
| `numpy` | 1.26.4 | Numerical operations for scoring |

### NLP & Quality Scoring

| Library | Version | Role |
|---------|---------|------|
| `sentence-transformers` | 3.0.1 | Semantic coherence scoring |
| `nltk` | 3.8.1 | Tokenization, n-gram computation |
| `spacy` | 3.7.4 | Linguistic annotation |
| `rouge-score` | 0.1.2 | ROUGE metrics for response quality |
| `bert-score` | 0.3.13 | Contextual semantic similarity |
| `textstat` | 0.7.3 | Readability and complexity metrics |

### Export & Dataset Tooling

| Library | Version | Role |
|---------|---------|------|
| `datasets` | 2.19.1 | HuggingFace-compatible dataset serialization |
| `huggingface-hub` | 0.23.0 | Optional push-to-hub for dataset sharing |

### Developer Experience

| Library | Version | Role |
|---------|---------|------|
| `click` | 8.1.7 | Clean CLI interfaces for all scripts |
| `loguru` | 0.7.2 | Structured, colored logging |
| `rich` | 13.7.1 | Terminal tables, progress bars, panels |
| `tqdm` | 4.66.4 | Progress tracking for bulk operations |
| `python-dotenv` | 1.0.1 | Environment variable management |
| `PyYAML` | 6.0.1 | Schema and config file parsing |

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Runtime secrets and thresholds |
| `config/annotation_schema.yaml` | Domain categories, quality dimensions, rejection criteria |
| `config/label_studio_template.xml` | Label Studio annotation interface |

---

## 5. Annotation Rubric

### Quality Dimensions (5-Point Scale)

```
Rating  Descriptor    Criteria
──────────────────────────────────────────────────────────────────────
  5     Excellent     Fully addresses intent, accurate, fluent, safe,
                      appropriately concise. No edits needed.
  4     Good          Minor imperfections; acceptable for SFT training.
  3     Acceptable    Noticeable issues but core info correct.
  2     Poor          Significant accuracy or coherence issues.
  1     Unacceptable  Reject — harmful, incoherent, or factually wrong.
```

### Weighted Quality Score Formula

```
Q = 0.30 × (helpfulness/5)
  + 0.25 × (accuracy/5)
  + 0.20 × (coherence/5)
  + 0.15 × (safety/5)
  + 0.10 × (conciseness/5)

Minimum passing score: Q ≥ 0.70
```

---

## 6. Output Schema Reference

### OpenAI Chat Format (recommended for fine-tuning)
```json
{
  "messages": [
    { "role": "user",      "content": "Explain gradient descent." },
    { "role": "assistant", "content": "Gradient descent is an optimization..." }
  ],
  "meta": {
    "task_id": 42,
    "source": "sample_pairs.jsonl",
    "annotations": {
      "helpfulness": 5, "accuracy": 4, "coherence": 5,
      "safety": 5, "conciseness": 4,
      "domain": "factual_qa",
      "verdict": "accept",
      "tags": ["verified", "gold_standard"]
    },
    "export_version": "v1.0",
    "exported_at": "2026-06-25T08:00:00Z"
  }
}
```

### Alpaca Format
```json
{ "instruction": "...", "input": "", "output": "...", "meta": {} }
```

### ShareGPT Format
```json
{ "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}] }
```

---

## 7. Project File Structure

```
local studio llm/
│
├── README.md                          # Quick start guide
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── PROCESS_AND_TECHSTACK.md           # This document
│
├── config/
│   ├── annotation_schema.yaml         # Rubric config, domains, tags
│   └── label_studio_template.xml      # Label Studio UI interface
│
├── scripts/
│   ├── setup_label_studio.py          # Initialize LS project
│   ├── ingest_raw_data.py             # Parse & upload raw data
│   ├── validate_dataset.py            # Quality gate filtering
│   ├── rubric_audit.py                # IAA + bias + gold extraction
│   └── export_jsonl.py                # Generate SFT-ready JSONL
│
└── data/
    ├── raw/                           # Input raw text files
    │   ├── sample_pairs.json
    │   └── sample_pairs.jsonl
    ├── annotated/                     # Post-annotation exports
    │   ├── validated_clean.json
    │   └── validated_rejected.json
    └── exports/                       # Final model-ready datasets
        ├── train_openai_v1.0_*.jsonl
        ├── val_openai_v1.0_*.jsonl
        └── test_openai_v1.0_*.jsonl
```

---

## 8. Key Design Decisions

**Programmatic Label Studio Setup** — All project creation is code-driven via the SDK (no manual UI clicks), ensuring reproducibility across environments and annotator teams.

**Schema-Driven Rubric** — The annotation rubric lives in `annotation_schema.yaml`, not hardcoded in Python. Adding a new quality dimension requires only a YAML edit and a `--force` rebuild.

**Multi-Schema JSONL Export** — The same validated dataset can be exported in Alpaca, ShareGPT, or OpenAI format, enabling direct use with LLaMA-Factory, Axolotl, or the OpenAI fine-tuning API without re-annotating.

**Full Provenance Metadata** — Every exported record carries `task_id`, `source_file`, all annotator scores, domain tag, and export version for complete dataset lineage tracing.

**Lightweight Coherence Scoring** — The validator uses bigram overlap as a fast, dependency-free proxy. For production, swap `check_coherence()` with `sentence-transformers` cosine similarity for semantic-level validation.

---

## 9. Metrics & Quality Gates

| Gate | Metric | Default Threshold |
|------|--------|------------------|
| Prompt length | Character count | 10 – 2048 chars |
| Response length | Character count | 20 – 4096 chars |
| Quality score | Weighted dim average | ≥ 0.70 |
| Coherence | Bigram Jaccard overlap | ≥ 0.65 |
| IAA target | Cohen's Kappa | ≥ 0.60 (Substantial) |
| Gold threshold | All dims per annotator | ≥ 4 / 5 |

All thresholds are configurable via `.env`.

---

## 10. Quickstart Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env   # set LABEL_STUDIO_API_KEY

# 3. Start Label Studio
label-studio start

# 4. Initialize annotation project
python scripts/setup_label_studio.py --project-name "My SFT Dataset"

# 5. Ingest raw data
python scripts/ingest_raw_data.py --input data/raw/
python scripts/ingest_raw_data.py --input data/raw/ --dry-run   # preview only

# 6. [Annotate via Label Studio UI → http://localhost:8080]

# 7. Validate annotated export
python scripts/validate_dataset.py --input data/annotated/export.json
python scripts/validate_dataset.py --input data/annotated/export.json --strict

# 8. Audit annotation consistency
python scripts/rubric_audit.py --input data/annotated/export.json --export-gold

# 9. Export to model-ready JSONL
python scripts/export_jsonl.py --schema openai --split            # train/val/test split
python scripts/export_jsonl.py --schema alpaca --filter-tag gold_standard
```

---

*LDIRC Pipeline — Local Data Instruction Response Curation Tool*
*Version: 1.0 | June 2026*
