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

`translate_pipeline.py` keeps the `| SourceFile: ...` suffix on the first output line (copied from the Chinese header if the model omits it), so you can map outputs back to the original filename.

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
- Optional: `LLM_COMPLETION_FACTS`, `LLM_COMPLETION_RELATIONS`, `LLM_COMPLETION_SCENES` (defaults 2300) for metadata JSONL stages — raise for verbose / reasoning models (see `.env.example`).
- Title line: `LLM_COMPLETION_TRANSLATE_TITLE` (default 256). Chunk caps: `LLM_TIMELINE_MAX_CHARS`, `LLM_METADATA_*_CHUNK_MAX_CHARS`, glossary row caps `LLM_GLOSSARY_*_MAX_ENTRIES`, etc. — see `.env.example` comments.

### Profiles: small context (~13k) vs large context (~100k)

`.env.example` documents **Profile A** (tight VRAM, ~13k context, e.g. Gemma) and **Profile B** (large context, e.g. Qwen 3.5 35B uncensored ~100k). If you raise `LLM_CONTEXT_TOKENS` for bigger translate chunks, also raise `LLM_COMPLETION_TRANSLATE` so the model is not cut off mid-chapter.

### Qwen-style translate + CJK line-fix (translate stage only)

For models that sometimes leak Chinese into Vietnamese lines:

- Use `--translate-prompt` (or env `TRANSLATE_PROMPT_FILE`) pointing at `prompts/translate_qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive.txt` (or your own template with `{glossary_block}` + `{user_input}`).
- Enable **`--translate-cjk-validate`** (or env `TRANSLATE_CJK_VALIDATE=1`). This runs **after** each translate chunk: rule-based detection (CJK ideographs in a VI line), then up to **`CJK_VALIDATE_RETRIES`** (default 3) LLM calls that receive **only the bad lines** (`zh` + faulty `vi`) and return JSONL fixes; good lines are **not** re-sent or re-translated.
- Optional: `--cjk-retry-prompt` / `CJK_RETRY_PROMPT_FILE`, `--completion-cjk-fix` / `LLM_COMPLETION_CJK_FIX`.
- The legacy **`validate.txt`** pass (`--no-validate` to disable) is **not** line-based: each chunk gets an extra LLM call with **full ZH + full VI**. If the model answers `FIXED_VI`, it must output the **entire chunk again** (second full completion). If parsing fails, the pipeline used to run **another full translate** + **validate** (up to **4 LLM calls per chunk**). That is separate from CJK line-fix and much more expensive.
  - To align with **line-only fixes**: use **`--no-validate`** + **`--translate-cjk-validate`** (CJK leak only).
  - To keep validate but skip the expensive re-translate loop: **`--no-validate-retry-whole-chunk`** or env **`VALIDATE_RETRY_WHOLE_CHUNK=0`** (keeps first VI if validate does not return PASS/FIXED_VI; then CJK can still run).
- **Glossary / timeline / metadata stages** use the **full chapter file** (including the `Chapter N - …` line) for chunking — no separate title/body split for extract.
- Those stages have **no** CJK line-fix (translate only).

`runs/*.jsonl` may include `stage` values like `translate_chunk_cjk` with `line_count_mismatch`, `cjk_retry_attempt`, `bad_line_count_*`, and `cjk_leak_unresolved` if some lines still contain CJK after retries.

### Prompt compliance harness (debugging format / `thinking` leaks)

Some local models wrap answers in `<thinking>…</thinking>` or similar; that used to break JSONL parsing. The pipeline now **sanitizes** those blocks before `parse_jsonl_objects` and `parse_validate_response`, and can **recover** inline JSON objects via a balanced-brace scan when lines are not clean JSONL.

To **re-test every prompt** on the first line-safe chunk of the default chapter (`chapters/C000-正文 序章：大魔王.txt`, includes the `Chapter …` line when present) without writing glossary/metadata outputs:

```powershell
python test_prompts_chapter.py --max-chars 4000
```

Artifacts go under `runs/prompt_tests/<UTC_timestamp>/`: each stage has `*.raw.txt` (exact model output) and `*.meta.json` (parse stats, timing). `summary.json` aggregates results.

- `--dry-run` — no LLM calls (only creates empty artifacts / summary is minimal).
- `--only glossary,cjk_retry` — run a subset.
- `--translate-prompt` — same as the main pipeline (e.g. Qwen aggressive file).

If `parse_ok_nonempty` is false for extract stages, open the matching `.raw.txt` and tighten prompts or server settings (temperature, template).

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

Example with Qwen aggressive prompt + CJK line-fix (no legacy `validate.txt`):

```powershell
python translate_pipeline.py --all --no-validate --translate-prompt translate_qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive.txt --translate-cjk-validate --context-tokens 100000
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
- `metadata/entity_facts.jsonl`: per-entity state/goal summaries. Extraction can emit parallel `*_vi` fields on `facts` and `evidence_vi` on each evidence (see `prompts/extract_entity_facts.txt`); `translate_metadata_vi.py` still fills any missing `*_vi` / `evidence_vi`.
- `metadata/relation_edges.jsonl`: edge/triple-like relations; each evidence should include `evidence_zh` + `evidence_vi` when extracting (`prompts/extract_relation_edges.txt`). `translate_metadata_vi.py` can still add missing `evidence_vi`.
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
- **LM Studio + “reasoning” models:** some APIs return an empty `message.content` and put everything in `message.reasoning_content`. The client now falls back to `reasoning_content` when `content` is blank. If the server reports `finish_reason: "length"` and JSONL is cut off, raise `LLM_COMPLETION_EXTRACT` / `LLM_COMPLETION_TRANSLATE` (and related limits) so the model can finish after its long thinking trace.
- **Official `openai` Python SDK (or `httpx`):** they are still HTTP + JSON to `/v1/chat/completions`. They do **not** merge `reasoning_content` into `content` for you — you would need the same fallback logic, or server-side settings so the final answer lands in `content`. Switching libraries does not fix model verbosity or token limits by itself.
- **`max_tokens` in the API:** this field is a **hard upper bound** on *new* completion tokens, not a target. The model stops when it is done (or hits the cap). It does not “pad” output to reach `max_tokens`. If you still want the server default instead, set **`LLM_OMIT_MAX_TOKENS=1`** in `.env` — then `max_tokens` is not sent (behavior depends on LM Studio; risks: longer generations, timeouts, or errors if the server requires the field).