#!/usr/bin/env python3
"""
Prompt compliance harness: one LLM call per prompt type on a chapter slice,
writes raw responses + meta under runs/prompt_tests/<timestamp>/.

Does not modify data/glossary.jsonl or translation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_client import client_from_env

from translate_pipeline import (
    BudgetConfig,
    Pipeline,
    _try_load_dotenv,
    build_line_chunks,
    chapter_header_line,
    fill_template,
    parse_chapter_header_line,
    parse_jsonl_objects,
    parse_validate_response,
    read_text,
    resolve_prompt_file,
)

_MIN_GLOSSARY_JSONL = "\n".join(
    [
        json.dumps(
            {
                "canonical_id": "mowang",
                "kind": "character",
                "names_zh": ["魔王"],
                "names_vi": ["Ma vương"],
                "evidences": [{"source_file": "dummy.txt", "evidence": "魔王"}],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "canonical_id": "yongzhe",
                "kind": "character",
                "names_zh": ["勇者"],
                "names_vi": ["Dũng sĩ"],
                "evidences": [{"source_file": "dummy.txt", "evidence": "勇者"}],
            },
            ensure_ascii=False,
        ),
    ]
)


def _system_msg() -> dict[str, str]:
    return {
        "role": "system",
        "content": "You follow instructions exactly. Never add markdown fences unless explicitly requested.",
    }


def _save_artifacts(out_dir: Path, name: str, raw: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.raw.txt").write_text(raw, encoding="utf-8", newline="\n")
    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    _try_load_dotenv()
    parser = argparse.ArgumentParser(description="Test all pipeline prompts against one chapter (LLM calls).")
    parser.add_argument(
        "--chapter",
        type=str,
        default="chapters/C000-正文 序章：大魔王.txt",
        help="Chapter path (relative to project root or absolute)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Max chars per line-safe chunk of the full chapter file (includes Chapter … line in chunk 0)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/prompt_tests",
        help="Base directory; a UTC timestamp subfolder is created",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: glossary,timeline,facts,relations,scenes,cjk_retry,validate,translate,translate_title "
        "(translate_title = isolated test of prompts/translate_title.txt; pipeline uses translate.txt on full chapter chunks)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM; write plan only")
    parser.add_argument("--prompts-dir", type=str, default="prompts")
    parser.add_argument("--translate-prompt", type=str, default="translate.txt")
    parser.add_argument("--cjk-retry-prompt", type=str, default="translate_retry_cjk_leak.txt")
    parser.add_argument("--context-tokens", type=int, default=int(os.environ.get("LLM_CONTEXT_TOKENS", "13000")))
    parser.add_argument("--completion-extract", type=int, default=int(os.environ.get("LLM_COMPLETION_EXTRACT", "1800")))
    parser.add_argument("--completion-translate", type=int, default=int(os.environ.get("LLM_COMPLETION_TRANSLATE", "4500")))
    parser.add_argument(
        "--completion-translate-title",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_TRANSLATE_TITLE", "256")),
    )
    parser.add_argument("--completion-validate", type=int, default=int(os.environ.get("LLM_COMPLETION_VALIDATE", "3200")))
    parser.add_argument("--completion-timeline", type=int, default=int(os.environ.get("LLM_COMPLETION_TIMELINE", "2500")))
    parser.add_argument("--completion-facts", type=int, default=int(os.environ.get("LLM_COMPLETION_FACTS", "2300")))
    parser.add_argument("--completion-relations", type=int, default=int(os.environ.get("LLM_COMPLETION_RELATIONS", "2300")))
    parser.add_argument("--completion-scenes", type=int, default=int(os.environ.get("LLM_COMPLETION_SCENES", "2300")))
    parser.add_argument("--completion-cjk-fix", type=int, default=int(os.environ.get("LLM_COMPLETION_CJK_FIX", "1536")))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ch_path = Path(args.chapter)
    if not ch_path.is_file():
        ch_path = root / args.chapter
    if not ch_path.is_file():
        print(f"Chapter not found: {args.chapter}", file=sys.stderr)
        return 2

    prompts_dir = (root / args.prompts_dir).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (root / args.out_dir / ts).resolve()

    only = {x.strip().lower() for x in args.only.split(",") if x.strip()} if args.only else set()

    def want(name: str) -> bool:
        return not only or name in only

    text = read_text(ch_path)
    header_line = chapter_header_line(text)
    source_file = ch_path.name
    chunks = build_line_chunks(text, max(args.max_chars, 64))
    chunk = chunks[0] if chunks else ""
    zh_lines = chunk.splitlines()
    parsed = parse_chapter_header_line(header_line)
    chapter_number = int(parsed[0]) if parsed else 0
    if chapter_number == 0:
        m = re.search(r"Chapter\s+(\d+)\s*-", header_line or "")
        chapter_number = int(m.group(1)) if m else 0

    translate_prompt_path = resolve_prompt_file(root, prompts_dir, args.translate_prompt)
    cjk_retry_prompt_path = resolve_prompt_file(root, prompts_dir, args.cjk_retry_prompt)

    glossary_path = root / "data" / "glossary.jsonl"
    if not glossary_path.is_file():
        glossary_path.parent.mkdir(parents=True, exist_ok=True)
        glossary_path.write_text("", encoding="utf-8")

    client = client_from_env()
    budget = BudgetConfig(
        context_tokens=args.context_tokens,
        completion_translate=args.completion_translate,
        completion_translate_title=args.completion_translate_title,
        completion_extract=args.completion_extract,
        completion_validate=args.completion_validate,
        completion_timeline=args.completion_timeline,
        completion_facts=args.completion_facts,
        completion_relations=args.completion_relations,
        completion_scenes=args.completion_scenes,
        completion_cjk_fix=args.completion_cjk_fix,
    )
    pipe = Pipeline(
        root=root,
        client=client,
        budget=budget,
        prompts_dir=prompts_dir,
        glossary_path=glossary_path,
        dry_run=False,
        validate_enabled=True,
        translate_header=True,
        timeline_enabled=True,
        timeline_path=root / "timeline" / "timeline_events.jsonl",
        timeline_max_chars=int(
            os.environ.get("LLM_TIMELINE_MAX_CHARS", os.environ.get("TIMELINE_MAX_CHARS", "9000"))
        ),
        facts_enabled=True,
        relations_enabled=True,
        scenes_enabled=True,
        translate_prompt_path=translate_prompt_path,
        translate_cjk_validate_enabled=False,
        cjk_retry_prompt_path=cjk_retry_prompt_path,
    )

    summary: list[dict] = []
    sys_m = _system_msg()

    def llm_chat(user: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, float]:
        if args.dry_run:
            return "", 0.0
        t0 = time.perf_counter()
        res = client.chat(
            [sys_m, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0
        return (res.content or ""), elapsed

    # --- glossary ---
    if want("glossary"):
        user = fill_template(
            pipe.prompt_extract,
            existing_glossary_block="(none)",
            source_chunk=chunk,
            source_file=source_file,
        )
        raw, elapsed = llm_chat(user, args.completion_extract, 0.2)
        n = len(parse_jsonl_objects(raw))
        ok = n > 0
        meta = {
            "prompt": "extract_glossary",
            "jsonl_object_count": n,
            "parse_ok_nonempty": ok,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_extract,
        }
        _save_artifacts(out_dir, "glossary", raw, meta)
        summary.append({"name": "glossary", **meta})

    # --- timeline ---
    if want("timeline"):
        user = fill_template(
            pipe.prompt_timeline,
            chapter_glossary_block=_MIN_GLOSSARY_JSONL,
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk=chunk,
        )
        raw, elapsed = llm_chat(user, args.completion_timeline, 0.2)
        n = len(parse_jsonl_objects(raw))
        meta = {
            "prompt": "extract_timeline_events",
            "jsonl_object_count": n,
            "parse_ok_nonempty": n > 0,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_timeline,
        }
        _save_artifacts(out_dir, "timeline", raw, meta)
        summary.append({"name": "timeline", **meta})

    # --- facts ---
    if want("facts"):
        user = fill_template(
            pipe.prompt_facts,
            chapter_glossary_block=_MIN_GLOSSARY_JSONL,
            timeline_events_block="(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk=chunk,
        )
        raw, elapsed = llm_chat(user, args.completion_facts, 0.2)
        n = len(parse_jsonl_objects(raw))
        meta = {
            "prompt": "extract_entity_facts",
            "jsonl_object_count": n,
            "parse_ok_nonempty": n > 0,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_facts,
        }
        _save_artifacts(out_dir, "facts", raw, meta)
        summary.append({"name": "facts", **meta})

    # --- relations ---
    if want("relations"):
        user = fill_template(
            pipe.prompt_relations,
            chapter_glossary_block=_MIN_GLOSSARY_JSONL,
            entity_facts_block="(none)",
            timeline_events_block="(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk=chunk,
        )
        raw, elapsed = llm_chat(user, args.completion_relations, 0.2)
        n = len(parse_jsonl_objects(raw))
        meta = {
            "prompt": "extract_relation_edges",
            "jsonl_object_count": n,
            "parse_ok_nonempty": n > 0,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_relations,
            "note": "empty JSONL is valid if model finds no edges",
        }
        _save_artifacts(out_dir, "relations", raw, meta)
        summary.append({"name": "relations", **meta})

    # --- scenes ---
    if want("scenes"):
        user = fill_template(
            pipe.prompt_scenes,
            chapter_glossary_block=_MIN_GLOSSARY_JSONL,
            timeline_events_block="(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk=chunk,
        )
        raw, elapsed = llm_chat(user, args.completion_scenes, 0.2)
        n = len(parse_jsonl_objects(raw))
        meta = {
            "prompt": "extract_scene_segments",
            "jsonl_object_count": n,
            "parse_ok_nonempty": n > 0,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_scenes,
        }
        _save_artifacts(out_dir, "scenes", raw, meta)
        summary.append({"name": "scenes", **meta})

    # --- cjk retry (synthetic bad line: VI == ZH to force CJK in VI) ---
    if want("cjk_retry"):
        li = 0
        z0 = zh_lines[li] if zh_lines else "魔王出现了。"
        bad_pairs = [{"line": li, "zh": z0, "vi": z0}]
        bad_jsonl = "\n".join(json.dumps(p, ensure_ascii=False) for p in bad_pairs)
        user = fill_template(
            pipe.prompt_cjk_retry,
            glossary_block=_MIN_GLOSSARY_JSONL,
            bad_lines_jsonl=bad_jsonl,
        )
        raw, elapsed = llm_chat(user, args.completion_cjk_fix, 0.05)
        objs = parse_jsonl_objects(raw)
        fixed = [
            o
            for o in objs
            if isinstance(o, dict) and "line" in o and "vi" in o and isinstance(o.get("vi"), str)
        ]
        meta = {
            "prompt": "translate_retry_cjk_leak",
            "jsonl_object_count": len(objs),
            "fix_objects_with_line_vi": len(fixed),
            "parse_ok_line_vi": len(fixed) > 0,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_cjk_fix,
        }
        _save_artifacts(out_dir, "cjk_retry", raw, meta)
        summary.append({"name": "cjk_retry", **meta})

    # --- validate ---
    if want("validate"):
        gloss = _MIN_GLOSSARY_JSONL
        vi_placeholder = "（Đây là bản dịch giả để test định dạng — this line is a test.）\n" + chunk[: min(800, len(chunk)) :]
        user = fill_template(
            pipe.prompt_validate,
            glossary_block=gloss,
            zh_chunk=chunk,
            vi_chunk=vi_placeholder,
        )
        raw, elapsed = llm_chat(user, args.completion_validate, 0.15)
        _fixed, ok = parse_validate_response(raw, vi_placeholder)
        meta = {
            "prompt": "validate",
            "parse_ok_pass_or_fixed": ok,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_validate,
        }
        _save_artifacts(out_dir, "validate", raw, meta)
        summary.append({"name": "validate", **meta})

    # --- translate body chunk ---
    if want("translate"):
        user = fill_template(
            pipe.prompt_translate,
            glossary_block=_MIN_GLOSSARY_JSONL,
            user_input=chunk,
        )
        raw, elapsed = llm_chat(user, args.completion_translate, 0.2)
        vi_lines = raw.splitlines()
        mismatch = len(zh_lines) != len(vi_lines)
        meta = {
            "prompt": "translate",
            "zh_line_count": len(zh_lines),
            "vi_line_count": len(vi_lines),
            "line_count_mismatch": mismatch,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_translate,
        }
        _save_artifacts(out_dir, "translate", raw, meta)
        summary.append({"name": "translate", **meta})

    # --- translate_title (prompt file only; main pipeline does not call this) ---
    if want("translate_title"):
        title_line = (
            header_line.strip()
            if header_line.strip().startswith("Chapter ")
            else f"Chapter {chapter_number} - 序章"
        )
        user = fill_template(
            pipe.prompt_translate_title,
            glossary_block="(none)",
            chapter_number=str(chapter_number),
            title_line=title_line,
        )
        raw, elapsed = llm_chat(user, args.completion_translate_title, 0.2)
        first = (raw.strip().splitlines() or [""])[0].strip().strip("`").strip()
        ok = bool(first) and first.lower().startswith("chapter ")
        meta = {
            "prompt": "translate_title",
            "first_line": first[:200],
            "starts_with_chapter": ok,
            "elapsed_s": round(elapsed, 3),
            "max_tokens": args.completion_translate_title,
        }
        _save_artifacts(out_dir, "translate_title", raw, meta)
        summary.append({"name": "translate_title", **meta})

    if args.dry_run:
        for row in summary:
            row["dry_run"] = True

    report = {
        "chapter": str(ch_path),
        "chunk_chars": len(chunk),
        "max_chars": args.max_chars,
        "utc": ts,
        "dry_run": args.dry_run,
        "results": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
