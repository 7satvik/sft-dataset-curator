# 🧠 LOCAL DATA INSTRUCTION RESPONSE CURATION (LDIRC) Tool

> A production-grade data annotation pipeline for converting raw unstructured text into structured Supervised Fine-Tuning (SFT) datasets using Label Studio.

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/setup_label_studio.py
python scripts/ingest_raw_data.py --input data/raw/
python scripts/validate_dataset.py --input data/annotated/
python scripts/export_jsonl.py --output data/exports/
```
