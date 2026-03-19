#!/usr/bin/env python3
"""
Translate existing metadata JSONL files (entity facts / relation edges / scenes)
to add Vietnamese fields, without re-extracting from chapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_client import client_from_env

# Reuse small helpers from the main pipeline.
from translate_pipeline import _try_load_dotenv, fill_template, strip_markdown_fences


def parse_jsonl_objects(content: str) -> list[dict]:
    content = strip_markdown_fences(content)
    out: list[dict] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def write_jsonl_objects(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(o, ensure_ascii=False) for o in objs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def translate_text(
    client,
    translate_prompt_tpl: str,
    system_content: str,
    text: str,
    glossary_block: str = "(none)",
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    user = fill_template(translate_prompt_tpl, glossary_block=glossary_block, user_input=text)
    msgs = [{"role": "system", "content": system_content}, {"role": "user", "content": user}]
    return client.chat(msgs, max_tokens=max_tokens, temperature=temperature).content.strip()


def main() -> int:
    _try_load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=str, default="prompts")
    parser.add_argument("--glossary", type=str, default="data/glossary.jsonl")

    parser.add_argument("--metadata-path-entity-facts", type=str, default="metadata/entity_facts.jsonl")
    parser.add_argument("--metadata-path-relation-edges", type=str, default="metadata/relation_edges.jsonl")
    parser.add_argument("--metadata-path-scenes", type=str, default="metadata/scenes.jsonl")

    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-missing", action="store_true", default=True)

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    prompts_dir = (root / args.prompts_dir).resolve()

    translate_prompt_tpl = (prompts_dir / "translate.txt").read_text(encoding="utf-8")
    client = client_from_env()

    # Keep consistent with translate_pipeline's system message.
    system_content = "You follow instructions exactly. Never add markdown fences unless explicitly requested."

    glossary_block = "(none)"
    # Optional: if you prefer glossary-grounding you can load/select relevant glossary,
    # but for metadata-short fields it's usually safe to keep (none).

    entity_facts_path = (root / args.metadata_path_entity_facts).resolve()
    relation_edges_path = (root / args.metadata_path_relation_edges).resolve()
    scenes_path = (root / args.metadata_path_scenes).resolve()

    # ---- entity_facts.jsonl ----
    if entity_facts_path.is_file():
        objs = parse_jsonl_objects(entity_facts_path.read_text(encoding="utf-8", errors="ignore"))
        changed = 0
        for obj in objs:
            facts = obj.get("facts")
            if isinstance(facts, dict):
                for k, v in list(facts.items()):
                    if not isinstance(v, str):
                        continue
                    if not v.strip():
                        continue
                    if k.endswith("_vi"):
                        continue
                    vi_key = f"{k}_vi"
                    if args.only_missing and isinstance(facts.get(vi_key), str) and facts[vi_key].strip():
                        continue
                    facts[vi_key] = translate_text(
                        client,
                        translate_prompt_tpl,
                        system_content,
                        v,
                        glossary_block=glossary_block,
                        max_tokens=args.max_tokens,
                    )
                    changed += 1

            evidences = obj.get("evidences")
            if isinstance(evidences, list):
                for ev in evidences:
                    if not isinstance(ev, dict):
                        continue
                    zh = ev.get("evidence_zh")
                    if not isinstance(zh, str) or not zh.strip():
                        continue
                    if args.only_missing and isinstance(ev.get("evidence_vi"), str) and ev["evidence_vi"].strip():
                        continue
                    ev["evidence_vi"] = translate_text(
                        client,
                        translate_prompt_tpl,
                        system_content,
                        zh,
                        glossary_block=glossary_block,
                        max_tokens=args.max_tokens,
                    )
                    changed += 1

        if changed and not args.dry_run:
            write_jsonl_objects(entity_facts_path, objs)
        print(f"entity_facts: changed={changed}, dry_run={args.dry_run}")
    else:
        print(f"entity_facts: missing {entity_facts_path}")

    # ---- relation_edges.jsonl ----
    if relation_edges_path.is_file():
        objs = parse_jsonl_objects(relation_edges_path.read_text(encoding="utf-8", errors="ignore"))
        changed = 0
        for obj in objs:
            evidences = obj.get("evidences")
            if not isinstance(evidences, list):
                continue
            for ev in evidences:
                if not isinstance(ev, dict):
                    continue
                zh = ev.get("evidence_zh")
                if not isinstance(zh, str) or not zh.strip():
                    continue
                if args.only_missing and isinstance(ev.get("evidence_vi"), str) and ev["evidence_vi"].strip():
                    continue
                ev["evidence_vi"] = translate_text(
                    client,
                    translate_prompt_tpl,
                    system_content,
                    zh,
                    glossary_block=glossary_block,
                    max_tokens=args.max_tokens,
                )
                changed += 1

        if changed and not args.dry_run:
            write_jsonl_objects(relation_edges_path, objs)
        print(f"relation_edges: changed={changed}, dry_run={args.dry_run}")
    else:
        print(f"relation_edges: missing {relation_edges_path}")

    # ---- scenes.jsonl ----
    if scenes_path.is_file():
        objs = parse_jsonl_objects(scenes_path.read_text(encoding="utf-8", errors="ignore"))
        changed = 0
        for obj in objs:
            title_zh = obj.get("scene_title_zh")
            if isinstance(title_zh, str) and title_zh.strip():
                if not (args.only_missing and isinstance(obj.get("scene_title_vi"), str) and obj["scene_title_vi"].strip()):
                    obj["scene_title_vi"] = translate_text(
                        client,
                        translate_prompt_tpl,
                        system_content,
                        title_zh,
                        glossary_block=glossary_block,
                        max_tokens=args.max_tokens,
                    )
                    changed += 1

            # Ensure scene_summary_vi exists (it is required by the extraction prompt, but keep defensive).
            if not (args.only_missing and isinstance(obj.get("scene_summary_vi"), str) and obj["scene_summary_vi"].strip()):
                summary_zh = obj.get("scene_summary_zh")
                if isinstance(summary_zh, str) and summary_zh.strip():
                    obj["scene_summary_vi"] = translate_text(
                        client,
                        translate_prompt_tpl,
                        system_content,
                        summary_zh,
                        glossary_block=glossary_block,
                        max_tokens=args.max_tokens,
                    )
                    changed += 1

        if changed and not args.dry_run:
            write_jsonl_objects(scenes_path, objs)
        print(f"scenes: changed={changed}, dry_run={args.dry_run}")
    else:
        print(f"scenes: missing {scenes_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

