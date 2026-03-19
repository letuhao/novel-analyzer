# Novel Analyzer (ZH -> VI + RAG Metadata)

Pipeline for:
1) Scraping/download chapters (separate script)
2) Translating ZH to VI with a glossary for consistent named entities
3) Extracting structured RAG metadata in JSONL:
   - `timeline/timeline_events.jsonl`
   - `metadata/entity_facts.jsonl`
   - `metadata/relation_edges.jsonl`
   - `metadata/scenes.jsonl`

It supports deterministic `*_id` and safe `--resume` for incremental runs.

## Project Layout

- `chapters/`: raw chapter text files (ZH), named like `C002-正文 第二章、魔王的無奈.txt`
- `data/`: persistent data
  - `data/glossary.jsonl`: entity glossary (canonical_id, aliases, evidences, ...)
- `prompts/`: LLM prompt templates
- `timeline/`: extracted timeline events (one JSONL file)
  - `timeline/timeline_events.jsonl`
- `metadata/`: extracted RAG metadata (JSONL files)
  - `metadata/entity_facts.jsonl`
  - `metadata/relation_edges.jsonl`
  - `metadata/scenes.jsonl`
- `translations/vi/`: translated chapter text output
- `runs/`: per-run debug logs (`*.jsonl`)
- `add_chapter_metadata.py`: prepends chapter header metadata to `chapters/*.txt`

## Header Format for Better RAG Referencing

`add_chapter_metadata.py` prepends a single first line header:

`Chapter {N} - {ZH title} | SourceFile: {filename}.txt`

`translate_pipeline.py` preserves the `| SourceFile: ...` suffix after translating the Vietnamese title, so you can reliably map translated text and metadata back to the original filename.

## Prerequisites

1) Python 3.10+ (recommended)
2) An OpenAI-compatible chat API (local LLM server / LM Studio / etc.)

The code uses `llm_client.py` which expects an OpenAI-like endpoint.
Typical usage:
- base URL: `http://localhost:1234`
- endpoint: `/v1/chat/completions`

## Configure LLM (env)

Edit `.env` (do NOT commit secrets).
An example is in `.env.example`.

You should set at least:
- LLM base URL
- API key (if your server requires it; otherwise can be blank)
- model name (e.g. `google/gemma-3-27b`)

## Step 1: Ensure Chapter Headers Exist

After you download/prepare chapters in `chapters/`, run:

```powershell
python add_chapter_metadata.py --chapters-dir chapters --force
```

This rewrites the first line so the pipeline can parse `Chapter N - ...`.

## Step 2: Run Translation + Metadata Extraction

For a single chapter:

```powershell
python translate_pipeline.py --chapter "chapters/C002-正文 第二章、魔王的無奈.txt" --no-validate --no-translate-header
```

For all chapters:

```powershell
python translate_pipeline.py --all --no-validate
```

Notes:
- `--resume` will skip already translated outputs (and metadata uses deterministic IDs, so re-running is safe).
- Timeline extraction and metadata stages run sequentially:
  1) glossary
  2) timeline
  3) entity facts
  4) relation edges
  5) scenes
  6) translation

## Step 3: (Optional) Translate Metadata to Vietnamese

By design, metadata extraction outputs ZH fields for some stages.
Run this script after you finish translating all chapters:

```powershell
python translate_metadata_vi.py
```

It adds Vietnamese fields like:
- `facts.state_vi`, `facts.goal_vi` (and `evidences[].evidence_vi`)
- `scene_title_vi` (if missing)

## Outputs You Can Index for RAG

Recommended ingestion targets:
- `timeline/timeline_events.jsonl`: event-centric summaries with `entities` and `evidences`
- `metadata/entity_facts.jsonl`: per-entity state/goal summaries (Vietnamese fields after `translate_metadata_vi.py`)
- `metadata/relation_edges.jsonl`: edge/triple-like relations between entities (Vietnamese evidence after translation)
- `metadata/scenes.jsonl`: scene/segment summaries with involved entities

## GitHub Setup (How to Publish)

1) Create a GitHub repository (empty).
2) Initialize git locally in this folder:

```powershell
git init
git add .
git commit -m "Initial commit: translation + RAG metadata pipeline"
```

3) Add your GitHub remote and push:

```powershell
git branch -M main
git remote add origin https://github.com/<your_user>/<your_repo>.git
git push -u origin main
```

Security:
- `.gitignore` already ignores `.env` and caches.
- Do not commit model secrets/keys.

## Troubleshooting

- If LLM calls fail, verify `.env` values and your server supports `/v1/chat/completions`.
- If headers don’t parse, rerun `add_chapter_metadata.py --force`.
- If metadata is empty, check `runs/*.jsonl` for the last successful stage.