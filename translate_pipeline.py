#!/usr/bin/env python3
"""
ZH -> VI translation pipeline with glossary extraction, chunked translation, and validation.
See plan: context budget (~13k), newline-safe chunking, JSONL glossary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_client import client_from_env


def _try_load_dotenv() -> None:
    p = Path(__file__).resolve().parent / ".env"
    if not p.is_file():
        return
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        return


def est_tokens(text: str) -> int:
    """Conservative estimate for CJK-heavy text."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_header_body(text: str) -> tuple[str, str]:
    """
    Legacy split: first line vs rest. Prefer using chapter_header_line() + full file text for pipeline passes.
    """
    lines = text.splitlines()
    if not lines:
        return "", ""
    if lines[0].strip().startswith("Chapter "):
        return lines[0], "\n".join(lines[1:]).lstrip("\n")
    return "", text


def chapter_header_line(text: str) -> str:
    """First line when it matches add_chapter_metadata format (Chapter N - …). Used only for chapter_number / metadata."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("Chapter "):
        return lines[0]
    return ""


def body_after_chapter_header_line(text: str) -> str:
    """Everything after the first line (for --no-translate-header: do not send line 1 to translate)."""
    lines = text.splitlines()
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[1:]).lstrip("\n")


# Metadata line from add_chapter_metadata.py: "Chapter 123 - 正文 ..."
CHAPTER_HEADER_RE = re.compile(r"^Chapter\s+(\d+)\s*-\s*(.+)$", re.IGNORECASE)


def _strip_sourcefile_suffix_from_title(title_rest: str) -> str:
    # New header format may append: "| SourceFile: <filename>".
    return re.sub(r"\s*\|\s*SourceFile:\s*.+$", "", title_rest or "").strip()


def parse_chapter_header_line(header: str) -> tuple[str, str] | None:
    """
    Returns (chapter_number_str, chinese_title_rest) if the line matches.
    """
    m = CHAPTER_HEADER_RE.match(header.strip())
    if not m:
        return None
    title_rest = _strip_sourcefile_suffix_from_title(m.group(2).strip())
    return m.group(1), title_rest


def ensure_vi_chapter_keeps_sourcefile_suffix(zh_chapter: str, vi_chapter: str) -> str:
    """
    add_chapter_metadata.py appends '| SourceFile: …' on the ZH header line.
    When the title is translated inside the main chunk (no dedicated header call), the model may drop it;
    copy the suffix from ZH line 1 onto VI line 1 if missing.
    """
    z_lines = zh_chapter.splitlines()
    v_lines = vi_chapter.splitlines()
    if not z_lines or not v_lines:
        return vi_chapter
    m = re.search(r"(\|\s*SourceFile:\s*.+)$", z_lines[0].strip(), flags=re.IGNORECASE)
    if not m:
        return vi_chapter
    suffix = m.group(1).strip()
    if re.search(r"\|\s*SourceFile:\s*", v_lines[0], flags=re.IGNORECASE):
        return vi_chapter
    v_lines[0] = f"{v_lines[0].rstrip()} {suffix}".rstrip()
    return "\n".join(v_lines)


def build_line_chunks(text: str, max_chunk_chars: int) -> list[str]:
    """
    Pack whole lines (split on '\\n') into chunks without breaking inside a line.
    If a single line exceeds max_chunk_chars, it becomes its own chunk (overflow).
    """
    if max_chunk_chars < 64:
        max_chunk_chars = 64
    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    cur_len = 0

    for line in lines:
        add_len = len(line) + (1 if buf else 0)
        if buf and cur_len + add_len > max_chunk_chars:
            chunks.append("\n".join(buf))
            buf = [line]
            cur_len = len(line)
        else:
            buf.append(line)
            cur_len += add_len

    if buf:
        chunks.append("\n".join(buf))
    return chunks


def strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


# Common "thinking" / reasoning wrappers leaked into message content by some local servers.
_LLM_THINKING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*think\s*>[\s\S]*?</\s*think\s*>", re.IGNORECASE),
    re.compile(r"<\s*redacted_reasoning\s*>[\s\S]*?</\s*redacted_reasoning\s*>", re.IGNORECASE),
    re.compile(r"<thinking>[\s\S]*?</thinking>", re.IGNORECASE),
    re.compile(r"<reasoning>[\s\S]*?</reasoning>", re.IGNORECASE),
    re.compile(r"```(?:think|thinking|reasoning)[\s\S]*?```", re.IGNORECASE),
)


def sanitize_llm_text_for_parse(content: str) -> str:
    """
    Strip thinking/reasoning blocks and trim whitespace before JSONL or PASS/FIXED_VI parsing.
    Does not guarantee valid JSON; see parse_jsonl_objects for recovery heuristics.
    """
    s = content or ""
    for pat in _LLM_THINKING_PATTERNS:
        s = pat.sub("", s)
    return s.strip()


def extract_json_dicts_balanced(text: str) -> list[dict]:
    """
    Find top-level {...} spans (string-aware) and json.loads each dict.
    Fallback when the model mixes prose with JSON or uses multi-line objects without JSONL newlines.
    """
    out: list[dict] = []
    s = text or ""
    i = 0
    n = len(s)
    while i < n:
        j = s.find("{", i)
        if j < 0:
            break
        depth = 0
        start = j
        k = j
        in_str = False
        esc = False
        quote = ""
        while k < n:
            ch = s[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
                k += 1
                continue
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                k += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : k + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        i = j + 1
                        break
                    if isinstance(obj, dict):
                        out.append(obj)
                    i = k + 1
                    break
            k += 1
        else:
            i = j + 1
    return out


def parse_jsonl_objects(content: str) -> list[dict]:
    content = sanitize_llm_text_for_parse(strip_markdown_fences(content))
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
    if not out:
        out = extract_json_dicts_balanced(content)
    return out


def normalize_canonical_id(cid: str) -> str:
    s = (cid or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def derive_canonical_id_from_names_zh(name_zh: str) -> str:
    """
    Deterministic ASCII-only fallback.
    Used only when the model canonical_id is missing/unknown and we didn't map by exact names_zh.
    """
    raw = (name_zh or "").strip()
    if not raw:
        return "unknown"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"zh_{h}"


def _dedupe_evidences(items: list[dict]) -> list[dict]:
    """Unique by (source_file, evidence) after strip; preserves first-seen order."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        sf = str(x.get("source_file", "")).strip()
        ev = str(x.get("evidence", "")).strip()
        if not sf and not ev:
            continue
        # Filter empty evidence to keep glossary clean.
        if not ev:
            continue
        key = (sf, ev)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source_file": sf, "evidence": ev})
    return out


def entry_to_evidences_list(obj: dict) -> list[dict]:
    """Collect evidences from new schema and legacy evidence/source_file fields."""
    raw: list[dict] = []
    evs = obj.get("evidences")
    if isinstance(evs, list):
        for x in evs:
            if isinstance(x, dict):
                raw.append(
                    {
                        "source_file": str(x.get("source_file", "")).strip(),
                        "evidence": str(x.get("evidence", "")).strip(),
                    }
                )
    # Legacy single fields (older glossary rows / model output)
    sf = obj.get("source_file")
    ev = obj.get("evidence")
    if isinstance(sf, str) and isinstance(ev, str) and (sf.strip() or ev.strip()):
        if ev.strip():
            raw.append({"source_file": sf.strip(), "evidence": ev.strip()})
    elif isinstance(sf, str) and sf.strip() and isinstance(ev, str) and ev.strip():
        raw.append({"source_file": sf.strip(), "evidence": ev.strip()})
    elif isinstance(ev, str) and ev.strip():
        raw.append({"source_file": "", "evidence": ev.strip()})
    return _dedupe_evidences(raw)


def finalize_glossary_entry(obj: dict) -> dict:
    """Single on-disk shape: evidences[] only; drop legacy keys."""
    e = dict(obj)
    merged = entry_to_evidences_list(e)
    if merged:
        e["evidences"] = merged
    else:
        e.pop("evidences", None)
    e.pop("evidence", None)
    e.pop("source_file", None)
    return e


def glossary_quality_score(obj: dict) -> int:
    evs = obj.get("evidences")
    if isinstance(evs, list):
        return sum(1 for x in evs if isinstance(x, dict) and str(x.get("evidence", "")).strip())
    # legacy fallback
    ev = obj.get("evidence")
    if isinstance(ev, str) and ev.strip():
        return 1
    return 0


def build_name_to_best_cid(glossary: dict[str, dict]) -> dict[str, str]:
    """
    Build mapping: exact names_zh surface form -> best canonical_id.
    Deterministic: higher evidence count wins; tie -> smaller canonical_id lexical.
    """
    name_to_best: dict[str, str] = {}
    name_to_best_score: dict[str, int] = {}

    for cid, obj in glossary.items():
        score = glossary_quality_score(obj)
        names = obj.get("names_zh") if isinstance(obj.get("names_zh"), list) else []
        for n in names:
            if not isinstance(n, str):
                continue
            t = n.strip()
            if not t:
                continue
            prev = name_to_best.get(t)
            if prev is None:
                name_to_best[t] = cid
                name_to_best_score[t] = score
            else:
                prev_score = name_to_best_score.get(t, 0)
                if score > prev_score or (score == prev_score and str(cid) < str(prev)):
                    name_to_best[t] = cid
                    name_to_best_score[t] = score

    return name_to_best


def normalize_glossary_by_names_zh(glossary: dict[str, dict]) -> dict[str, dict]:
    """
    Dedupe glossary entities deterministically by exact `names_zh` surface forms.

    Strategy:
    - For each `names_zh` string, pick a best canonical_id:
        higher evidences count wins, then lexicographical canonical_id.
    - Merge all other entries that contain those surface forms into the best entry.
    """
    # Build name_zh -> best_cid index.
    name_to_best: dict[str, str] = {}
    name_to_best_score: dict[str, int] = {}

    for cid, obj in glossary.items():
        score = glossary_quality_score(obj)
        names = obj.get("names_zh") if isinstance(obj.get("names_zh"), list) else []
        for n in names:
            if not isinstance(n, str):
                continue
            t = n.strip()
            if not t:
                continue
            prev = name_to_best.get(t)
            if prev is None:
                name_to_best[t] = cid
                name_to_best_score[t] = score
            else:
                prev_score = name_to_best_score.get(t, 0)
                if score > prev_score or (score == prev_score and str(cid) < str(prev)):
                    name_to_best[t] = cid
                    name_to_best_score[t] = score

    if not name_to_best:
        # Nothing to dedupe.
        return glossary

    def choose_target_cid(obj: dict, current_cid: str) -> str:
        names = obj.get("names_zh") if isinstance(obj.get("names_zh"), list) else []
        candidates: set[str] = set()
        for n in names:
            if not isinstance(n, str):
                continue
            t = n.strip()
            if not t:
                continue
            target = name_to_best.get(t)
            if target:
                candidates.add(target)
        if not candidates:
            return current_cid
        if len(candidates) == 1:
            return next(iter(candidates))
        # If multiple candidates, pick best by quality among those candidates.
        best = None
        best_score = -1
        for c in candidates:
            q = glossary_quality_score(glossary.get(c, {}))
            if q > best_score or (q == best_score and (best is None or str(c) < str(best))):
                best_score = q
                best = c
        return str(best) if best is not None else current_cid

    # Merge into targets.
    new_glossary: dict[str, dict] = {}
    touched: set[str] = set()

    for cid, obj in glossary.items():
        target = choose_target_cid(obj, cid)
        if target not in new_glossary:
            new_glossary[target] = dict(obj)
        else:
            merge_entry(new_glossary[target], obj)
        touched.add(target)

    # Finalize shape.
    for c in list(touched):
        new_glossary[c] = finalize_glossary_entry(new_glossary[c])

    return new_glossary


def attach_default_source_to_incoming_entry(obj: dict, default_source_file: str) -> None:
    """Fill missing source_file on evidences[] or legacy fields (mutates obj)."""
    evs = obj.get("evidences")
    if isinstance(evs, list):
        for rec in evs:
            if isinstance(rec, dict) and not str(rec.get("source_file", "")).strip():
                rec["source_file"] = default_source_file
    if isinstance(obj.get("evidence"), str) and obj["evidence"].strip():
        obj.setdefault("source_file", default_source_file)


def merge_entry(dst: dict, src: dict) -> dict:
    def uniq_extend(key: str) -> None:
        a = dst.get(key)
        b = src.get(key)
        if isinstance(a, list) and isinstance(b, list):
            seen = set()
            merged: list = []
            for x in a + b:
                if not isinstance(x, str):
                    continue
                t = x.strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                merged.append(t)
            dst[key] = merged
        elif isinstance(b, list) and not a:
            dst[key] = b

    uniq_extend("names_zh")
    uniq_extend("names_vi")

    da = dst.get("aliases") if isinstance(dst.get("aliases"), dict) else {}
    sa = src.get("aliases") if isinstance(src.get("aliases"), dict) else {}
    if sa:
        out_aliases: dict[str, list[str]] = {}
        for k in ("zh", "vi"):
            xs = []
            if isinstance(da.get(k), list):
                xs.extend([str(x).strip() for x in da[k] if str(x).strip()])
            if isinstance(sa.get(k), list):
                xs.extend([str(x).strip() for x in sa[k] if str(x).strip()])
            seen = set()
            merged = []
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    merged.append(x)
            if merged:
                out_aliases[k] = merged
        if out_aliases:
            dst["aliases"] = out_aliases

    # Evidences: merge + dedupe by (source_file, evidence) for cross-chapter traceability
    combined = entry_to_evidences_list(dst) + entry_to_evidences_list(src)
    deduped = _dedupe_evidences(combined)
    if deduped:
        dst["evidences"] = deduped
    dst.pop("evidence", None)
    dst.pop("source_file", None)

    if src.get("kind") and not dst.get("kind"):
        dst["kind"] = src["kind"]
    return dst


def load_glossary(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        cid = normalize_canonical_id(str(obj.get("canonical_id", "")))
        if not cid or cid == "unknown":
            continue
        obj["canonical_id"] = cid
        if cid in out:
            merge_entry(out[cid], obj)
        else:
            out[cid] = obj
    # Normalize legacy rows after merge
    for cid, obj in out.items():
        out[cid] = finalize_glossary_entry(obj)

    # Dedupe by exact names_zh (deterministic canonical reuse)
    out = normalize_glossary_by_names_zh(out)
    for cid, obj in out.items():
        out[cid] = finalize_glossary_entry(obj)
    return out


def save_glossary(path: Path, glossary: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cid in sorted(glossary.keys()):
        clean = finalize_glossary_entry(glossary[cid])
        lines.append(json.dumps(clean, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def append_reject(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def glossary_to_lines(glossary: dict[str, dict]) -> list[str]:
    return [json.dumps(glossary[k], ensure_ascii=False) for k in sorted(glossary.keys())]


def trim_block_to_tokens(block: str, max_tokens: int) -> str:
    if est_tokens(block) <= max_tokens:
        return block
    # Trim by whole lines from the end
    lines = block.splitlines()
    while lines and est_tokens("\n".join(lines)) > max_tokens:
        lines.pop()
    return "\n".join(lines)


def select_relevant_glossary(
    glossary: dict[str, dict],
    chunk_text: str,
    max_tokens: int,
    max_entries: int,
) -> str:
    scored: list[tuple[int, str, dict]] = []
    for cid, obj in glossary.items():
        names = obj.get("names_zh") if isinstance(obj.get("names_zh"), list) else []
        best = 0
        for n in names:
            if not isinstance(n, str):
                continue
            t = n.strip()
            if not t:
                continue
            if t in chunk_text:
                best = max(best, len(t))
        if best > 0:
            scored.append((best, cid, obj))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked: list[dict] = []
    for _, _, obj in scored[:max_entries]:
        picked.append(obj)
    block = "\n".join(json.dumps(x, ensure_ascii=False) for x in picked)
    return trim_block_to_tokens(block, max_tokens)


def select_existing_for_extract(
    glossary: dict[str, dict],
    chunk_text: str,
    max_tokens: int,
    max_entries: int,
) -> str:
    # Prefer relevant entries; if room remains, include some recent/smaller set is optional v1: relevant only
    return select_relevant_glossary(glossary, chunk_text, max_tokens=max_tokens, max_entries=max_entries)


def build_chapter_glossary_block(
    glossary: dict[str, dict],
    chapter_text: str,
    max_tokens: int,
    max_entries: int,
    pinned_names_zh: list[str] | None = None,
) -> str:
    """
    Build a stable mini-glossary for the entire chapter.

    Scoring:
    - Count occurrences of each names_zh surface form in chapter_text.
    - Score = sum(occurrences * max(1, len(name))).
    """
    pinned_names_zh = pinned_names_zh or []
    pinned_names_zh_set = set(pinned_names_zh)

    scored: list[tuple[int, str, dict]] = []
    pinned: list[dict] = []

    for cid, obj in glossary.items():
        names = obj.get("names_zh") if isinstance(obj.get("names_zh"), list) else []
        best_score = 0
        hit_pinned = False
        for n in names:
            if not isinstance(n, str):
                continue
            t = n.strip()
            if not t:
                continue
            if t in pinned_names_zh_set and t in chapter_text:
                hit_pinned = True
            # Conservative occurrence count for substring
            occ = chapter_text.count(t)
            if occ <= 0:
                continue
            best_score += occ * max(1, len(t))

        if best_score > 0:
            scored.append((best_score, cid, obj))
        if hit_pinned:
            pinned.append(obj)

    # Ensure pinned are always included.
    picked: list[dict] = []
    seen_cid: set[str] = set()

    for obj in pinned:
        cid = str(obj.get("canonical_id", "")).strip()
        if not cid:
            # fallback: find by reference is expensive; skip pinned without cid
            continue
        if cid in seen_cid:
            continue
        seen_cid.add(cid)
        picked.append(obj)

    # Fill the rest by score.
    scored.sort(key=lambda x: (-x[0], x[1]))
    for _, cid, obj in scored:
        if cid in seen_cid:
            continue
        seen_cid.add(cid)
        picked.append(obj)
        if len(picked) >= max_entries:
            break

    # If no scored entries but pinned exists, still try to inject something.
    if not picked:
        # Fall back to substring-based selection (simpler, may return small set)
        return select_relevant_glossary(
            glossary,
            chapter_text,
            max_tokens=max_tokens,
            max_entries=max_entries,
        )

    block = "\n".join(json.dumps(x, ensure_ascii=False) for x in picked)
    return trim_block_to_tokens(block, max_tokens)


@dataclass
class BudgetConfig:
    context_tokens: int = 13000
    safety_tokens: int = 400
    completion_translate: int = 4500
    completion_translate_title: int = 256
    completion_extract: int = 1800
    completion_validate: int = 3200
    completion_timeline: int = 2500
    completion_facts: int = 2300
    completion_relations: int = 2300
    completion_scenes: int = 2300
    completion_cjk_fix: int = 1536
    glossary_inject_tokens: int = 1200
    glossary_max_entries: int = 40
    existing_glossary_cap_tokens: int = 900
    # Title-line translate: glossary snippet + completion budget
    title_glossary_inject_cap: int = 800
    title_glossary_max_entries: int = 20
    # Caps on chapter glossary rows (min with glossary_max_entries in code)
    glossary_merge_max_entries: int = 25
    glossary_timeline_max_entries: int = 40
    glossary_metadata_max_entries: int = 45
    # Upper bounds on Chinese body chars per metadata chunk (after budget-derived cap)
    metadata_facts_chunk_max_chars: int = 6000
    metadata_relations_chunk_max_chars: int = 7000
    metadata_scenes_chunk_max_chars: int = 7000


def fill_template(tpl: str, **kwargs: str) -> str:
    out = tpl
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    missing = re.findall(r"\{([a-zA-Z0-9_]+)\}", out)
    if missing:
        raise ValueError(f"Missing template keys: {missing}")
    return out


# CJK Unified Ideographs (basic leak detection for Vietnamese output)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def vi_line_contains_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


def cjk_leak_bad_line_indices(zh_lines: list[str], vi_lines: list[str]) -> tuple[list[int], bool]:
    """
    Returns (bad_indices, line_count_mismatch).
    bad_indices: 0-based lines in vi that still contain CJK when a vi line exists.
    """
    mismatch = len(zh_lines) != len(vi_lines)
    bad: list[int] = []
    n = min(len(zh_lines), len(vi_lines))
    for i in range(n):
        if vi_line_contains_cjk(vi_lines[i]):
            bad.append(i)
    for i in range(n, len(vi_lines)):
        if vi_line_contains_cjk(vi_lines[i]):
            bad.append(i)
    return bad, mismatch


def apply_cjk_fix_objects_to_vi_lines(
    vi_lines: list[str],
    objs: list[dict],
    allowed_indices: set[int],
) -> int:
    """Apply JSONL fix objects {"line": int, "vi": str}. Extends vi_lines if needed. Returns count applied."""
    applied = 0
    for o in objs:
        if not isinstance(o, dict):
            continue
        try:
            li = int(o.get("line", -1))
        except (TypeError, ValueError):
            continue
        if li not in allowed_indices:
            continue
        vi = o.get("vi")
        if not isinstance(vi, str):
            continue
        while len(vi_lines) <= li:
            vi_lines.append("")
        vi_lines[li] = vi.strip()
        applied += 1
    return applied


def parse_validate_response(validator_output: str, vi_original: str) -> tuple[str, bool]:
    t = sanitize_llm_text_for_parse(strip_markdown_fences(validator_output))
    if not t:
        return vi_original, False
    lines = t.splitlines()
    for idx, raw in enumerate(lines):
        first = raw.strip().strip("*`").strip()
        if not first:
            continue
        if first == "PASS":
            return vi_original, True
        if first == "FIXED_VI":
            fixed = "\n".join(lines[idx + 1 :])
            return fixed.strip(), True
    return vi_original, False


def compute_timeline_event_id(chapter_number: int, event_title_zh: str, entities: list[str]) -> str:
    ent = "|".join(sorted({str(x).strip() for x in entities if isinstance(x, str) and str(x).strip()}))
    base = f"{chapter_number}|{event_title_zh.strip()}|{ent}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def dedupe_timeline_evidences(evidences: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for rec in evidences:
        if not isinstance(rec, dict):
            continue
        sf = str(rec.get("source_file", "")).strip()
        ev = str(rec.get("evidence_zh") if rec.get("evidence_zh") is not None else rec.get("evidence", "")).strip()
        if not sf or not ev:
            continue
        key = (sf, ev)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source_file": sf, "evidence_zh": ev})
    return out


class TimelineStore:
    def __init__(self, timeline_path: Path) -> None:
        self.timeline_path = timeline_path
        self.events_by_id: dict[str, dict] = {}
        self.order: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.timeline_path.is_file():
            return
        try:
            objs = parse_jsonl_objects(self.timeline_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return

        for obj in objs:
            if not isinstance(obj, dict):
                continue
            try:
                chapter_number = int(obj.get("chapter_number", 0))
            except Exception:
                continue
            title = str(obj.get("event_title_zh", "")).strip()
            entities = obj.get("entities", [])
            if not title:
                continue
            eid = compute_timeline_event_id(chapter_number, title, entities if isinstance(entities, list) else [])
            if eid in self.events_by_id:
                # Merge if already exists.
                self._merge_into(self.events_by_id[eid], obj)
            else:
                normalized = self._normalize_entry(obj, chapter_number, title)
                self.events_by_id[eid] = normalized
                self.order.append(eid)

    def _normalize_entry(self, obj: dict, chapter_number: int, title: str) -> dict:
        entities = obj.get("entities", [])
        entities_norm: list[str] = []
        if isinstance(entities, list):
            for x in entities:
                if isinstance(x, str) and x.strip() and x.strip() not in entities_norm:
                    entities_norm.append(x.strip())

        evidences = obj.get("evidences", [])
        if not isinstance(evidences, list):
            evidences = []
        evidences_norm = dedupe_timeline_evidences(evidences)

        return {
            "event_title_zh": title,
            "event_summary_zh": str(obj.get("event_summary_zh", "")).strip(),
            "event_summary_vi": str(obj.get("event_summary_vi", "")).strip(),
            "entities": entities_norm,
            "evidences": evidences_norm,
            "chapter_number": chapter_number,
        }

    def _merge_into(self, old: dict, new: dict) -> None:
        new_entities = new.get("entities", [])
        if isinstance(new_entities, list):
            for x in new_entities:
                if isinstance(x, str) and x.strip() and x.strip() not in old.get("entities", []):
                    old["entities"].append(x.strip())

        old_evs = old.get("evidences", [])
        merged_evs = dedupe_timeline_evidences(old_evs + (new.get("evidences", []) if isinstance(new.get("evidences"), list) else []))
        old["evidences"] = merged_evs

        # Update summaries if new provides richer info.
        old_sz_vi = len(str(old.get("event_summary_vi", "")))
        new_sz_vi = len(str(new.get("event_summary_vi", "")))
        if new_sz_vi > old_sz_vi and new_sz_vi > 0:
            old["event_summary_vi"] = str(new.get("event_summary_vi", "")).strip()

        old_sz_zh = len(str(old.get("event_summary_zh", "")))
        new_sz_zh = len(str(new.get("event_summary_zh", "")))
        if new_sz_zh > old_sz_zh and new_sz_zh > 0:
            old["event_summary_zh"] = str(new.get("event_summary_zh", "")).strip()

    def upsert(self, event_obj: dict) -> str | None:
        if not isinstance(event_obj, dict):
            return None
        try:
            chapter_number = int(event_obj.get("chapter_number", 0))
        except Exception:
            return None
        title = str(event_obj.get("event_title_zh", "")).strip()
        if not title:
            return None
        entities = event_obj.get("entities", [])
        ent_list = entities if isinstance(entities, list) else []
        eid = compute_timeline_event_id(chapter_number, title, ent_list)

        if eid in self.events_by_id:
            self._merge_into(self.events_by_id[eid], event_obj)
        else:
            self.events_by_id[eid] = self._normalize_entry(event_obj, chapter_number, title)
            self.order.append(eid)
        return eid

    def flush(self) -> None:
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for eid in self.order:
            lines.append(json.dumps(self.events_by_id[eid], ensure_ascii=False))
        self.timeline_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def _dedupe_evidence_pairs(evidences: list[dict], evidence_field: str = "evidence_zh") -> list[dict]:
    """
    Dedupe by (source_file, primary evidence text). Preserve extra fields (e.g. evidence_vi).
    """
    merged_by_key: dict[tuple[str, str], dict] = {}
    for rec in evidences:
        if not isinstance(rec, dict):
            continue
        sf = str(rec.get("source_file", "")).strip()
        ev = str(rec.get(evidence_field, "") if rec.get(evidence_field, "") is not None else rec.get("evidence", "")).strip()
        if not sf or not ev:
            continue
        key = (sf, ev)
        if key not in merged_by_key:
            row = {k: v for k, v in rec.items()}
            row["source_file"] = sf
            row[evidence_field] = ev
            merged_by_key[key] = row
        else:
            acc = merged_by_key[key]
            for k, v in rec.items():
                if k in ("source_file", evidence_field, "evidence"):
                    continue
                if v is None:
                    continue
                if not str(v).strip():
                    continue
                prev = acc.get(k)
                if prev is None or (isinstance(prev, str) and not prev.strip()):
                    acc[k] = v
    return list(merged_by_key.values())


def _facts_payload_for_dedupe_hash(facts: dict) -> dict:
    """
    Strip *_vi sibling keys so fact_id stays stable when Vietnamese is present or added later.
    """
    if not isinstance(facts, dict):
        return {}
    return {k: v for k, v in facts.items() if isinstance(k, str) and not k.endswith("_vi")}


def compute_entity_fact_id(chapter_number: int, entity_id: str, facts: dict) -> tuple[str, str]:
    """
    Returns (fact_id, fact_key) deterministic for dedupe.
    """
    # fact_key is a hash of the facts payload (ZH / non-_vi keys only).
    facts_obj = _facts_payload_for_dedupe_hash(facts if isinstance(facts, dict) else {})
    fact_key = hashlib.sha1(json.dumps(facts_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    base = f"{chapter_number}|{entity_id}|{fact_key}"
    fact_id = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return fact_id, fact_key


class EntityFactsStore:
    def __init__(self, facts_path: Path) -> None:
        self.facts_path = facts_path
        self.facts_by_id: dict[str, dict] = {}
        self.order: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.facts_path.is_file():
            return
        try:
            objs = parse_jsonl_objects(self.facts_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            fid = str(obj.get("fact_id", "")).strip()
            if fid:
                self.facts_by_id[fid] = obj
                if fid not in self.order:
                    self.order.append(fid)
                continue
            # fallback compute if missing
            try:
                chapter_number = int(obj.get("chapter_number", 0))
            except Exception:
                continue
            entity_id = str(obj.get("entity_id", "")).strip()
            facts = obj.get("facts", {})
            if not entity_id:
                continue
            computed_id, _ = compute_entity_fact_id(chapter_number, entity_id, facts if isinstance(facts, dict) else {})
            self.facts_by_id[computed_id] = obj
            self.order.append(computed_id)

    def upsert(self, fact_obj: dict) -> str | None:
        if not isinstance(fact_obj, dict):
            return None
        try:
            chapter_number = int(fact_obj.get("chapter_number", 0))
        except Exception:
            return None
        entity_id = str(fact_obj.get("entity_id", "")).strip()
        if not entity_id:
            return None
        facts = fact_obj.get("facts", {})
        if not isinstance(facts, dict):
            facts = {}
        evidences = fact_obj.get("evidences", [])
        evidences_norm = _dedupe_evidence_pairs(evidences if isinstance(evidences, list) else [])

        fid, fkey = compute_entity_fact_id(chapter_number, entity_id, facts)
        if fid in self.facts_by_id:
            old = self.facts_by_id[fid]
            # merge evidences
            old_evs = old.get("evidences", [])
            merged = _dedupe_evidence_pairs((old_evs if isinstance(old_evs, list) else []) + evidences_norm)
            old["evidences"] = merged
            # keep facts if richer
            old_facts = old.get("facts", {})
            if isinstance(old_facts, dict) and isinstance(facts, dict):
                # Add missing keys from new (including *_vi when older row had only ZH).
                for k, v in facts.items():
                    prev = old_facts.get(k)
                    if k not in old_facts or prev is None or (isinstance(prev, str) and not str(prev).strip()):
                        old_facts[k] = v
                old["facts"] = old_facts
            self.facts_by_id[fid] = old
        else:
            self.facts_by_id[fid] = {
                "chapter_number": chapter_number,
                "entity_id": entity_id,
                "facts": facts,
                "evidences": evidences_norm,
                "fact_key": fkey,
                "fact_id": fid,
            }
            self.order.append(fid)
        return fid

    def flush(self) -> None:
        self.facts_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for fid in self.order:
            lines.append(json.dumps(self.facts_by_id[fid], ensure_ascii=False))
        self.facts_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def compute_relation_edge_id(
    chapter_number: int,
    source_entity: str,
    relation_type: str,
    target_entity: str,
) -> str:
    base = f"{chapter_number}|{source_entity}|{relation_type}|{target_entity}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


class RelationEdgesStore:
    def __init__(self, edges_path: Path) -> None:
        self.edges_path = edges_path
        self.edges_by_id: dict[str, dict] = {}
        self.order: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.edges_path.is_file():
            return
        try:
            objs = parse_jsonl_objects(self.edges_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            eid = str(obj.get("edge_id", "")).strip()
            if not eid:
                continue
            self.edges_by_id[eid] = obj
            if eid not in self.order:
                self.order.append(eid)

    def upsert(self, edge_obj: dict) -> str | None:
        if not isinstance(edge_obj, dict):
            return None
        try:
            chapter_number = int(edge_obj.get("chapter_number", 0))
        except Exception:
            return None
        source_entity = str(edge_obj.get("source_entity", "")).strip()
        target_entity = str(edge_obj.get("target_entity", "")).strip()
        relation_type = str(edge_obj.get("relation_type", "")).strip()
        if not source_entity or not target_entity or not relation_type:
            return None

        evidences = edge_obj.get("evidences", [])
        evidences_norm = _dedupe_evidence_pairs(evidences if isinstance(evidences, list) else [])

        eid = compute_relation_edge_id(chapter_number, source_entity, relation_type, target_entity)

        if eid in self.edges_by_id:
            old = self.edges_by_id[eid]
            old_evs = old.get("evidences", [])
            old["evidences"] = _dedupe_evidence_pairs(
                (old_evs if isinstance(old_evs, list) else []) + evidences_norm
            )

            # Optional summaries: keep the longer one.
            for k in ("summary_zh", "summary_vi"):
                ov = str(old.get(k, "") or "").strip()
                nv = str(edge_obj.get(k, "") or "").strip()
                if nv and len(nv) > len(ov):
                    old[k] = nv
            self.edges_by_id[eid] = old
        else:
            self.edges_by_id[eid] = {
                "chapter_number": chapter_number,
                "source_entity": source_entity,
                "relation_type": relation_type,
                "target_entity": target_entity,
                "evidences": evidences_norm,
                "edge_id": eid,
            }
            # Optional summaries.
            for k in ("summary_zh", "summary_vi"):
                if k in edge_obj:
                    sv = str(edge_obj.get(k, "") or "").strip()
                    if sv:
                        self.edges_by_id[eid][k] = sv
            self.order.append(eid)

        return eid

    def flush(self) -> None:
        self.edges_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for eid in self.order:
            lines.append(json.dumps(self.edges_by_id[eid], ensure_ascii=False))
        self.edges_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def compute_scene_id(chapter_number: int, scene_title_zh: str) -> str:
    base = f"{chapter_number}|{scene_title_zh.strip()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


class ScenesStore:
    def __init__(self, scenes_path: Path) -> None:
        self.scenes_path = scenes_path
        self.scenes_by_id: dict[str, dict] = {}
        self.order: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.scenes_path.is_file():
            return
        try:
            objs = parse_jsonl_objects(self.scenes_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            sid = str(obj.get("scene_id", "")).strip()
            if sid:
                self.scenes_by_id[sid] = obj
                if sid not in self.order:
                    self.order.append(sid)

    def upsert(self, scene_obj: dict) -> str | None:
        if not isinstance(scene_obj, dict):
            return None
        try:
            chapter_number = int(scene_obj.get("chapter_number", 0))
        except Exception:
            return None
        scene_title_zh = str(scene_obj.get("scene_title_zh", "")).strip()
        if not scene_title_zh:
            return None

        entities = scene_obj.get("entities", [])
        entities_norm: list[str] = []
        if isinstance(entities, list):
            for x in entities:
                if isinstance(x, str):
                    t = x.strip()
                    if t and t not in entities_norm:
                        entities_norm.append(t)

        evidences = scene_obj.get("evidences", [])
        evidences_norm = _dedupe_evidence_pairs(evidences if isinstance(evidences, list) else [])

        sid = compute_scene_id(chapter_number, scene_title_zh)

        if sid in self.scenes_by_id:
            old = self.scenes_by_id[sid]
            # merge entities
            old_ents = old.get("entities", [])
            if isinstance(old_ents, list):
                for e in entities_norm:
                    if e and e not in old_ents:
                        old_ents.append(e)
                old["entities"] = old_ents
            else:
                old["entities"] = entities_norm

            old["evidences"] = _dedupe_evidence_pairs(
                (old.get("evidences", []) if isinstance(old.get("evidences", []), list) else []) + evidences_norm
            )

            # update summaries if new provides longer content
            for k in ("scene_summary_zh", "scene_summary_vi"):
                ov = str(old.get(k, "") or "").strip()
                nv = str(scene_obj.get(k, "") or "").strip()
                if nv and len(nv) > len(ov):
                    old[k] = nv

            self.scenes_by_id[sid] = old
        else:
            new_entry = {
                "chapter_number": chapter_number,
                "scene_title_zh": scene_title_zh,
                "scene_summary_zh": str(scene_obj.get("scene_summary_zh", "")).strip(),
                "entities": entities_norm,
                "evidences": evidences_norm,
                "scene_id": sid,
            }
            sv = str(scene_obj.get("scene_summary_vi", "") or "").strip()
            if sv:
                new_entry["scene_summary_vi"] = sv
            self.scenes_by_id[sid] = new_entry
            self.order.append(sid)

        return sid

    def flush(self) -> None:
        self.scenes_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for sid in self.order:
            lines.append(json.dumps(self.scenes_by_id[sid], ensure_ascii=False))
        self.scenes_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


class Pipeline:
    def __init__(
        self,
        root: Path,
        client,
        budget: BudgetConfig,
        prompts_dir: Path,
        glossary_path: Path,
        dry_run: bool,
        validate_enabled: bool,
        translate_header: bool,
        timeline_enabled: bool,
        timeline_path: Path,
        timeline_max_chars: int,
        facts_enabled: bool = True,
        entity_facts_path: Path | None = None,
        relations_enabled: bool = True,
        relation_edges_path: Path | None = None,
        scenes_enabled: bool = True,
        scenes_path: Path | None = None,
        translate_prompt_path: Path | None = None,
        translate_cjk_validate_enabled: bool = False,
        cjk_validate_retries: int = 3,
        cjk_retry_prompt_path: Path | None = None,
        validate_retry_whole_chunk: bool = True,
    ) -> None:
        self.root = root
        self.client = client
        self.budget = budget
        self.prompts_dir = prompts_dir
        self.glossary_path = glossary_path
        self.dry_run = dry_run
        self.validate_enabled = validate_enabled
        self.validate_retry_whole_chunk = validate_retry_whole_chunk
        self.translate_header = translate_header
        self.translate_cjk_validate_enabled = translate_cjk_validate_enabled
        self.cjk_validate_retries = max(0, int(cjk_validate_retries))

        self.prompt_extract = (prompts_dir / "extract_glossary.txt").read_text(encoding="utf-8")
        tp = translate_prompt_path or (prompts_dir / "translate.txt")
        self.prompt_translate = tp.read_text(encoding="utf-8")
        self.prompt_validate = (prompts_dir / "validate.txt").read_text(encoding="utf-8")
        self.prompt_translate_title = (prompts_dir / "translate_title.txt").read_text(encoding="utf-8")
        self.prompt_timeline = (prompts_dir / "extract_timeline_events.txt").read_text(encoding="utf-8")
        self.prompt_facts = (prompts_dir / "extract_entity_facts.txt").read_text(encoding="utf-8")
        self.prompt_relations = (prompts_dir / "extract_relation_edges.txt").read_text(encoding="utf-8")
        self.prompt_scenes = (prompts_dir / "extract_scene_segments.txt").read_text(encoding="utf-8")
        crp = cjk_retry_prompt_path or (prompts_dir / "translate_retry_cjk_leak.txt")
        self.prompt_cjk_retry = crp.read_text(encoding="utf-8")

        self.timeline_enabled = timeline_enabled
        self.timeline_path = timeline_path
        self.timeline_max_chars = timeline_max_chars
        self.timeline_store = TimelineStore(self.timeline_path) if self.timeline_enabled else None

        # Metadata extraction outputs (can be disabled via CLI).
        self.facts_store = (
            EntityFactsStore(entity_facts_path or (root / "metadata" / "entity_facts.jsonl"))
            if facts_enabled
            else None
        )
        self.relations_store = (
            RelationEdgesStore(relation_edges_path or (root / "metadata" / "relation_edges.jsonl"))
            if relations_enabled
            else None
        )
        self.scenes_store = (
            ScenesStore(scenes_path or (root / "metadata" / "scenes.jsonl"))
            if scenes_enabled
            else None
        )

    def _system_msg(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": "You follow instructions exactly. Never add markdown fences unless explicitly requested.",
        }

    def run_cjk_line_fix_loop(
        self,
        zh_text: str,
        vi_text: str,
        glossary_block: str,
        log,
        *,
        chunk_index: int,
        stage: str = "translate_chunk_cjk",
    ) -> str:
        """
        Rule-based CJK leak detection per line; up to cjk_validate_retries LLM calls
        that only receive the bad (zh, vi) pairs. Does not re-translate the whole chunk.
        """
        if not self.translate_cjk_validate_enabled or self.dry_run:
            return vi_text
        zh_lines = zh_text.splitlines()
        vi_lines = vi_text.splitlines()
        bad, mismatch = cjk_leak_bad_line_indices(zh_lines, vi_lines)
        if mismatch:
            log(
                {
                    "stage": stage,
                    "chunk_index": chunk_index,
                    "line_count_mismatch": True,
                    "zh_line_count": len(zh_lines),
                    "vi_line_count": len(vi_lines),
                }
            )
        attempt = 0
        while bad and attempt < self.cjk_validate_retries:
            attempt += 1
            allowed = set(bad)
            pairs: list[dict] = []
            for i in sorted(bad):
                zhi = zh_lines[i] if i < len(zh_lines) else ""
                vii = vi_lines[i] if i < len(vi_lines) else ""
                pairs.append({"line": i, "zh": zhi, "vi": vii})
            bad_jsonl = "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs)
            user = fill_template(
                self.prompt_cjk_retry,
                glossary_block=glossary_block or "(none)",
                bad_lines_jsonl=bad_jsonl,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(
                msgs, max_tokens=self.budget.completion_cjk_fix, temperature=0.05
            ).content
            objs = parse_jsonl_objects(res)
            applied = apply_cjk_fix_objects_to_vi_lines(vi_lines, objs, allowed)
            bad_after, _ = cjk_leak_bad_line_indices(zh_lines, vi_lines)
            log(
                {
                    "stage": stage,
                    "chunk_index": chunk_index,
                    "cjk_retry_attempt": attempt,
                    "bad_line_count_before": len(bad),
                    "fix_objects_applied": applied,
                    "bad_line_count_after": len(bad_after),
                }
            )
            bad = bad_after
        if bad:
            log(
                {
                    "stage": stage,
                    "chunk_index": chunk_index,
                    "warn": "cjk_leak_unresolved",
                    "remaining_bad_lines": bad[:24],
                    "remaining_bad_count": len(bad),
                }
            )
        return "\n".join(vi_lines)

    def compute_chunk_limits(self) -> tuple[int, int, int]:
        """
        Returns (max_chars_extract, max_chars_translate, max_chars_validate_bound)
        validate bound is used to cap translate chunks when validation is enabled.
        """
        b = self.budget

        extract_shell = fill_template(
            self.prompt_extract,
            existing_glossary_block="",
            source_chunk="",
            source_file="X.txt",
        )
        translate_shell = fill_template(
            self.prompt_translate,
            glossary_block="",
            user_input="",
        )
        validate_shell = fill_template(
            self.prompt_validate,
            glossary_block="",
            zh_chunk="",
            vi_chunk="",
        )

        extract_base = est_tokens(extract_shell) + est_tokens(self._system_msg()["content"])
        translate_base = est_tokens(translate_shell) + est_tokens(self._system_msg()["content"])
        validate_base = est_tokens(validate_shell) + est_tokens(self._system_msg()["content"])

        # Worst-case glossary blocks for sizing
        gloss_pad = b.glossary_inject_tokens
        exist_pad = b.existing_glossary_cap_tokens

        extract_avail = b.context_tokens - b.safety_tokens - b.completion_extract - extract_base - exist_pad
        translate_avail = b.context_tokens - b.safety_tokens - b.completion_translate - translate_base - gloss_pad

        max_chars_extract = max(256, extract_avail * 3)
        max_chars_translate = max(256, translate_avail * 3)

        # Validator must fit ZH+VI (~2x chunk) in the same context
        validate_avail = b.context_tokens - b.safety_tokens - b.completion_validate - validate_base - gloss_pad
        pair_chunk_tokens = max(128, validate_avail // 2)
        max_chars_validate_pair = max(256, pair_chunk_tokens * 3)

        if self.validate_enabled:
            max_chars_translate = min(max_chars_translate, max_chars_validate_pair)

        return max_chars_extract, max_chars_translate, max_chars_validate_pair

    def run_glossary_pass(self, source_file: str, chapter_text: str, log) -> None:
        glossary = load_glossary(self.glossary_path)
        name_to_best_cid = build_name_to_best_cid(glossary)
        max_ce, _, _ = self.compute_chunk_limits()
        chunks = build_line_chunks(chapter_text, max_ce)
        log({"stage": "glossary", "chunks": len(chunks), "max_chars": max_ce})
        if self.dry_run:
            return

        # Build a stable "existing glossary" block for the ENTIRE chapter.
        # This makes canonical_id reuse more consistent across chunks.
        pinned = ["伊斯坦莎", "勇者", "魔王"]
        existing_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.existing_glossary_cap_tokens,
            max_entries=min(self.budget.glossary_max_entries, self.budget.glossary_merge_max_entries),
            pinned_names_zh=pinned,
        )

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_extract,
                existing_glossary_block=existing_glossary_block or "(none)",
                source_chunk=ch,
                source_file=source_file,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(msgs, max_tokens=self.budget.completion_extract, temperature=0.1)
            objs = parse_jsonl_objects(res.content)
            if not objs:
                append_reject(
                    self.glossary_path.parent / "glossary_rejects.jsonl",
                    {
                        "source_file": source_file,
                        "chunk_index": idx,
                        "finish_reason": getattr(res, "finish_reason", None),
                        "raw": res.content[:4000],
                    },
                )
                continue
            touched: set[str] = set()
            for obj in objs:
                names_zh = obj.get("names_zh")
                candidates: set[str] = set()
                if isinstance(names_zh, list):
                    for n in names_zh:
                        if isinstance(n, str):
                            t = n.strip()
                            if t and t in name_to_best_cid:
                                candidates.add(name_to_best_cid[t])

                cid_model = normalize_canonical_id(str(obj.get("canonical_id", "")))
                cid_model = cid_model if cid_model and cid_model != "unknown" else ""

                if candidates:
                    # Deterministic pick: best evidence count wins; tie by lexical.
                    best = None
                    best_score = -1
                    for c in candidates:
                        q = glossary_quality_score(glossary.get(c, {}))
                        if q > best_score or (q == best_score and (best is None or str(c) < str(best))):
                            best_score = q
                            best = c
                    cid = str(best)
                else:
                    if cid_model:
                        cid = cid_model
                    else:
                        # Fallback: deterministic hash-based canonical_id from the first names_zh.
                        first_name = ""
                        if isinstance(names_zh, list) and names_zh:
                            first_name = str(names_zh[0] or "")
                        cid = derive_canonical_id_from_names_zh(first_name)

                if not cid:
                    continue

                obj["canonical_id"] = cid
                attach_default_source_to_incoming_entry(obj, source_file)
                obj.setdefault("source_file", source_file)
                if cid in glossary:
                    merge_entry(glossary[cid], obj)
                else:
                    glossary[cid] = obj
                touched.add(cid)
            # Normalize to enforce names_zh->canonical_id consistency.
            glossary = normalize_glossary_by_names_zh(glossary)
            for cid in touched:
                if cid in glossary:
                    glossary[cid] = finalize_glossary_entry(glossary[cid])
            save_glossary(self.glossary_path, glossary)
            log({"stage": "glossary_chunk", "index": idx, "merged": len(objs)})

    def run_translate_pass(
        self,
        source_file: str,
        header_line: str,
        chapter_text: str,
        log,
    ) -> str:
        glossary = load_glossary(self.glossary_path)
        _, max_ct, _ = self.compute_chunk_limits()
        # Full chapter text in chunks (title line included in chunk 0) — no separate title LLM call.
        # With --no-translate-header, only translate body_after_chapter_header_line; keep header_line ZH.
        if self.translate_header or not header_line:
            translate_source = chapter_text
        else:
            translate_source = body_after_chapter_header_line(chapter_text)

        chunks = build_line_chunks(translate_source, max_ct)
        log(
            {
                "stage": "translate",
                "chunks": len(chunks),
                "max_chars": max_ct,
                "unified_chapter_text": True,
                "translate_header": self.translate_header,
            }
        )

        vi_chunks: list[str] = []

        if self.dry_run:
            if self.translate_header or not header_line:
                return "\n\n".join(f"[DRY-RUN]{c[:40]}..." for c in chunks)
            return header_line + "\n\n" + "\n\n".join(f"[DRY-RUN]{c[:40]}..." for c in chunks)

        # Build a stable mini-glossary for the entire chapter and inject it
        # into every translate/validate call (not per-chunk).
        pinned = ["伊斯坦莎", "勇者", "魔王"]
        chapter_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.glossary_inject_tokens,
            max_entries=self.budget.glossary_max_entries,
            pinned_names_zh=pinned,
        )

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_translate,
                glossary_block=chapter_glossary_block or "(none)",
                user_input=ch,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            vi = self.client.chat(msgs, max_tokens=self.budget.completion_translate, temperature=0.25).content.strip()

            # Legacy validate (prompts/validate.txt): sends FULL zh_chunk + FULL vi_chunk every time.
            # - If model returns PASS: one extra LLM call; VI unchanged.
            # - If FIXED_VI: same call but model must emit the ENTIRE chunk again (high completion cost).
            # - If parse fails and validate_retry_whole_chunk: +1 full re-translate + +1 full validate (very expensive).
            # For line-only fixes and lower cost, use --no-validate and --translate-cjk-validate instead.
            if self.validate_enabled:
                vb = chapter_glossary_block

                def _validate_current(vi_text: str) -> tuple[str, bool]:
                    vuser_local = fill_template(
                        self.prompt_validate,
                        glossary_block=vb or "(none)",
                        zh_chunk=ch,
                        vi_chunk=vi_text,
                    )
                    vmsgs_local = [self._system_msg(), {"role": "user", "content": vuser_local}]
                    vout_local = self.client.chat(
                        vmsgs_local, max_tokens=self.budget.completion_validate, temperature=0.0
                    ).content
                    return parse_validate_response(vout_local, vi_text)

                vi2, ok = _validate_current(vi)
                if ok:
                    vi = vi2
                elif self.validate_retry_whole_chunk:
                    # one retry translate with short hint
                    hint = (
                        "Your previous translation failed validation. "
                        "Improve fidelity and glossary consistency. Output ONLY Vietnamese.\n\n"
                    )
                    user2 = hint + user
                    msgs2 = [self._system_msg(), {"role": "user", "content": user2}]
                    vi = self.client.chat(msgs2, max_tokens=self.budget.completion_translate, temperature=0.15).content.strip()
                    vi3, ok2 = _validate_current(vi)
                    if ok2:
                        vi = vi3
                else:
                    log(
                        {
                            "stage": "translate_chunk",
                            "index": idx,
                            "warn": "validate_unparsed_or_fail_keep_vi",
                            "validate_retry_whole_chunk": False,
                        }
                    )

            if self.translate_cjk_validate_enabled:
                vi = self.run_cjk_line_fix_loop(
                    ch,
                    vi,
                    chapter_glossary_block or "(none)",
                    log,
                    chunk_index=idx,
                    stage="translate_chunk_cjk",
                )

            vi_chunks.append(vi)
            log({"stage": "translate_chunk", "index": idx, "vi_chars": len(vi)})

        body_vi = "\n".join(vi_chunks)
        if not self.translate_header and header_line:
            return header_line + "\n\n" + body_vi
        return ensure_vi_chapter_keeps_sourcefile_suffix(chapter_text, body_vi)

    def compute_timeline_chunk_max_chars(
        self,
        chapter_glossary_block: str,
        chapter_number: int,
        source_file: str,
    ) -> int:
        # Budget-aware cap using the model's own prompt size estimate.
        base_user = fill_template(
            self.prompt_timeline,
            chapter_glossary_block=chapter_glossary_block or "(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk="",
        )
        sys_tokens = est_tokens(self._system_msg()["content"])
        base_tokens = est_tokens(base_user) + sys_tokens
        avail = self.budget.context_tokens - self.budget.safety_tokens - self.budget.completion_timeline - base_tokens
        max_chars = max(256, int(avail * 3))
        if self.timeline_max_chars and self.timeline_max_chars > 0:
            max_chars = min(max_chars, self.timeline_max_chars)
        return max_chars

    def run_timeline_pass(self, source_file: str, header_line: str, chapter_text: str, log) -> None:
        if not self.timeline_enabled or not self.timeline_store:
            return

        # Extract chapter number from metadata line: "Chapter N - ..."
        parsed = parse_chapter_header_line(header_line)
        if parsed:
            chapter_number = int(parsed[0])
        else:
            m = re.search(r"Chapter\s+(\d+)\s*-", header_line or "")
            chapter_number = int(m.group(1)) if m else 0

        glossary = load_glossary(self.glossary_path)

        pinned = ["伊斯坦莎", "勇者", "魔王"]
        chapter_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.glossary_inject_tokens,
            max_entries=min(self.budget.glossary_max_entries, self.budget.glossary_timeline_max_entries),
            pinned_names_zh=pinned,
        )

        max_ct = self.compute_timeline_chunk_max_chars(
            chapter_glossary_block=chapter_glossary_block,
            chapter_number=chapter_number,
            source_file=source_file,
        )
        chunks = build_line_chunks(chapter_text, max_ct)
        log({"stage": "timeline", "chunks": len(chunks), "max_chars": max_ct})

        if self.dry_run:
            return

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_timeline,
                chapter_glossary_block=chapter_glossary_block or "(none)",
                chapter_number=str(chapter_number),
                source_file=source_file,
                source_chunk=ch,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(msgs, max_tokens=self.budget.completion_timeline, temperature=0.2)
            objs = parse_jsonl_objects(res.content)
            if not objs:
                continue

            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                event_title_zh = str(obj.get("event_title_zh", "")).strip()
                if not event_title_zh:
                    continue

                entities = obj.get("entities", [])
                entities_norm: list[str] = []
                if isinstance(entities, list):
                    for x in entities:
                        if isinstance(x, str) and x.strip() and x.strip() not in entities_norm:
                            entities_norm.append(x.strip())

                evidences = obj.get("evidences", [])
                evidences_norm = dedupe_timeline_evidences(evidences if isinstance(evidences, list) else [])

                event_obj = {
                    "event_title_zh": event_title_zh,
                    "event_summary_zh": str(obj.get("event_summary_zh", "")).strip(),
                    "event_summary_vi": str(obj.get("event_summary_vi", "")).strip(),
                    "entities": entities_norm,
                    "evidences": evidences_norm,
                    "chapter_number": chapter_number,
                }
                self.timeline_store.upsert(event_obj)

            log({"stage": "timeline_chunk", "index": idx, "events_seen": len(objs)})

        # Flush once per chapter.
        self.timeline_store.flush()

    def run_facts_pass(self, source_file: str, header_line: str, chapter_text: str, log) -> None:
        # Only run if we have timeline context (entity facts should be extracted after timeline).
        if not self.facts_store:
            return
        if not self.timeline_store:
            # Still attempt facts extraction without timeline injection.
            timeline_events_block = "(none)"
            chapter_number = 0
        else:
            parsed = parse_chapter_header_line(header_line)
            if parsed:
                chapter_number = int(parsed[0])
            else:
                m = re.search(r"Chapter\s+(\d+)\s*-", header_line or "")
                chapter_number = int(m.group(1)) if m else 0

            # Build a compact timeline block for this chapter to ground facts.
            timeline_events: list[dict] = []
            for eid in self.timeline_store.order:
                ev = self.timeline_store.events_by_id.get(eid) if isinstance(self.timeline_store.events_by_id, dict) else None
                if not isinstance(ev, dict):
                    continue
                if int(ev.get("chapter_number", 0)) != chapter_number:
                    continue
                timeline_events.append(
                    {
                        "event_title_zh": ev.get("event_title_zh", ""),
                        "event_summary_zh": ev.get("event_summary_zh", ""),
                        "entities": ev.get("entities", []),
                    }
                )
            timeline_events_block = "\n".join(json.dumps(x, ensure_ascii=False) for x in timeline_events[:10]) or "(none)"

        glossary = load_glossary(self.glossary_path)
        pinned = ["伊斯坦莎", "勇者", "魔王"]
        chapter_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.glossary_inject_tokens,
            max_entries=min(self.budget.glossary_max_entries, self.budget.glossary_metadata_max_entries),
            pinned_names_zh=pinned,
        )

        # Budget-aware chunk cap for facts extraction.
        base_user = fill_template(
            self.prompt_facts,
            chapter_glossary_block=chapter_glossary_block or "(none)",
            timeline_events_block=timeline_events_block or "(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk="",
        )
        sys_tokens = est_tokens(self._system_msg()["content"])
        base_tokens = est_tokens(base_user) + sys_tokens
        avail = self.budget.context_tokens - self.budget.safety_tokens - self.budget.completion_facts - base_tokens
        max_ct = max(256, int(avail * 3))
        max_ct = min(max_ct, self.budget.metadata_facts_chunk_max_chars)

        chunks = build_line_chunks(chapter_text, max_ct)
        log({"stage": "entity_facts", "chunks": len(chunks), "max_chars": max_ct})

        if self.dry_run:
            return

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_facts,
                chapter_glossary_block=chapter_glossary_block or "(none)",
                timeline_events_block=timeline_events_block or "(none)",
                chapter_number=str(chapter_number),
                source_file=source_file,
                source_chunk=ch,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(msgs, max_tokens=self.budget.completion_facts, temperature=0.2).content
            objs = parse_jsonl_objects(res)
            if not objs:
                continue

            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                # Canonicalize chapter_number/entity_id types in output.
                try:
                    obj["chapter_number"] = int(obj.get("chapter_number", chapter_number))
                except Exception:
                    obj["chapter_number"] = chapter_number
                if "entity_id" in obj:
                    obj["entity_id"] = str(obj.get("entity_id", "")).strip()

                self.facts_store.upsert(obj)

            log({"stage": "entity_facts_chunk", "index": idx, "events_seen": len(objs)})

        self.facts_store.flush()

    def run_relations_pass(self, source_file: str, header_line: str, chapter_text: str, log) -> None:
        glossary = load_glossary(self.glossary_path)

        parsed = parse_chapter_header_line(header_line)
        if parsed:
            chapter_number = int(parsed[0])
        else:
            m = re.search(r"Chapter\s+(\d+)\s*-", header_line or "")
            chapter_number = int(m.group(1)) if m else 0

        if not self.relations_store:
            return

        # Build stable chapter glossary block and extract canonical_id list.
        pinned = ["伊斯坦莎", "勇者", "魔王"]
        chapter_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.glossary_inject_tokens,
            max_entries=min(self.budget.glossary_max_entries, self.budget.glossary_metadata_max_entries),
            pinned_names_zh=pinned,
        )
        entities_in_chapter: set[str] = set()
        for raw in (chapter_glossary_block or "").splitlines():
            if not raw.strip():
                continue
            try:
                o = json.loads(raw)
            except Exception:
                continue
            cid = str(o.get("canonical_id", "")).strip()
            if cid:
                entities_in_chapter.add(cid)

        # Build chapter timeline block (compact).
        timeline_events: list[dict] = []
        if self.timeline_store and hasattr(self.timeline_store, "events_by_id"):
            for eid in self.timeline_store.order:
                ev = self.timeline_store.events_by_id.get(eid)
                if not isinstance(ev, dict):
                    continue
                if int(ev.get("chapter_number", 0)) != chapter_number:
                    continue
                timeline_events.append(
                    {
                        "event_title_zh": ev.get("event_title_zh", ""),
                        "event_summary_zh": ev.get("event_summary_zh", ""),
                        "entities": ev.get("entities", []),
                    }
                )
        timeline_events_block = "\n".join(json.dumps(x, ensure_ascii=False) for x in timeline_events[:8]) or "(none)"

        # Build entity facts block for entities present in this chapter.
        facts_items: list[dict] = []
        if self.facts_store and isinstance(self.facts_store.facts_by_id, dict):
            for fid, obj in self.facts_store.facts_by_id.items():
                if not isinstance(obj, dict):
                    continue
                if int(obj.get("chapter_number", 0)) != chapter_number:
                    continue
                ent = str(obj.get("entity_id", "")).strip()
                if ent and ent in entities_in_chapter:
                    facts_items.append(
                        {
                            "entity_id": ent,
                            "facts": obj.get("facts", {}),
                            "evidences": obj.get("evidences", []),
                        }
                    )
        entity_facts_block = "\n".join(json.dumps(x, ensure_ascii=False) for x in facts_items[:30]) or "(none)"

        base_user = fill_template(
            self.prompt_relations,
            chapter_glossary_block=chapter_glossary_block or "(none)",
            entity_facts_block=entity_facts_block or "(none)",
            timeline_events_block=timeline_events_block or "(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk="",
        )
        sys_tokens = est_tokens(self._system_msg()["content"])
        base_tokens = est_tokens(base_user) + sys_tokens
        avail = self.budget.context_tokens - self.budget.safety_tokens - self.budget.completion_relations - base_tokens
        max_ct = max(256, int(avail * 3))
        max_ct = min(max_ct, self.budget.metadata_relations_chunk_max_chars)

        chunks = build_line_chunks(chapter_text, max_ct)
        log({"stage": "relation_edges", "chunks": len(chunks), "max_chars": max_ct})

        if self.dry_run:
            return

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_relations,
                chapter_glossary_block=chapter_glossary_block or "(none)",
                entity_facts_block=entity_facts_block or "(none)",
                timeline_events_block=timeline_events_block or "(none)",
                chapter_number=str(chapter_number),
                source_file=source_file,
                source_chunk=ch,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(msgs, max_tokens=self.budget.completion_relations, temperature=0.2).content
            objs = parse_jsonl_objects(res)
            if not objs:
                continue
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                # Basic canonicalization
                if "chapter_number" in obj:
                    try:
                        obj["chapter_number"] = int(obj["chapter_number"])
                    except Exception:
                        obj["chapter_number"] = chapter_number
                self.relations_store.upsert(obj)
            log({"stage": "relation_edges_chunk", "index": idx, "edges_seen": len(objs)})

        self.relations_store.flush()

    def run_scene_pass(self, source_file: str, header_line: str, chapter_text: str, log) -> None:
        glossary = load_glossary(self.glossary_path)

        parsed = parse_chapter_header_line(header_line)
        if parsed:
            chapter_number = int(parsed[0])
        else:
            m = re.search(r"Chapter\s+(\d+)\s*-", header_line or "")
            chapter_number = int(m.group(1)) if m else 0

        if not self.scenes_store:
            return

        # Build stable chapter glossary block.
        pinned = ["伊斯坦莎", "勇者", "魔王"]
        chapter_glossary_block = build_chapter_glossary_block(
            glossary,
            chapter_text,
            max_tokens=self.budget.glossary_inject_tokens,
            max_entries=min(self.budget.glossary_max_entries, self.budget.glossary_metadata_max_entries),
            pinned_names_zh=pinned,
        )

        # Build compact timeline block for grounding.
        timeline_events: list[dict] = []
        if self.timeline_store and hasattr(self.timeline_store, "events_by_id"):
            for eid in self.timeline_store.order:
                ev = self.timeline_store.events_by_id.get(eid)
                if not isinstance(ev, dict):
                    continue
                if int(ev.get("chapter_number", 0)) != chapter_number:
                    continue
                timeline_events.append(
                    {
                        "event_title_zh": ev.get("event_title_zh", ""),
                        "event_summary_zh": ev.get("event_summary_zh", ""),
                        "entities": ev.get("entities", []),
                    }
                )
        timeline_events_block = "\n".join(json.dumps(x, ensure_ascii=False) for x in timeline_events[:8]) or "(none)"

        base_user = fill_template(
            self.prompt_scenes,
            chapter_glossary_block=chapter_glossary_block or "(none)",
            timeline_events_block=timeline_events_block or "(none)",
            chapter_number=str(chapter_number),
            source_file=source_file,
            source_chunk="",
        )
        sys_tokens = est_tokens(self._system_msg()["content"])
        base_tokens = est_tokens(base_user) + sys_tokens
        avail = self.budget.context_tokens - self.budget.safety_tokens - self.budget.completion_scenes - base_tokens
        max_ct = max(256, int(avail * 3))
        max_ct = min(max_ct, self.budget.metadata_scenes_chunk_max_chars)

        chunks = build_line_chunks(chapter_text, max_ct)
        log({"stage": "scenes", "chunks": len(chunks), "max_chars": max_ct})

        if self.dry_run:
            return

        for idx, ch in enumerate(chunks):
            user = fill_template(
                self.prompt_scenes,
                chapter_glossary_block=chapter_glossary_block or "(none)",
                timeline_events_block=timeline_events_block or "(none)",
                chapter_number=str(chapter_number),
                source_file=source_file,
                source_chunk=ch,
            )
            msgs = [self._system_msg(), {"role": "user", "content": user}]
            res = self.client.chat(msgs, max_tokens=self.budget.completion_scenes, temperature=0.2).content
            objs = parse_jsonl_objects(res)
            if not objs:
                continue
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                if "chapter_number" in obj:
                    try:
                        obj["chapter_number"] = int(obj["chapter_number"])
                    except Exception:
                        obj["chapter_number"] = chapter_number
                else:
                    obj["chapter_number"] = chapter_number
                self.scenes_store.upsert(obj)
            log({"stage": "scenes_chunk", "index": idx, "scenes_seen": len(objs)})

        self.scenes_store.flush()

    def process_file(self, chapter_path: Path, out_dir: Path, run_log_path: Path | None) -> None:
        chapter_text = read_text(chapter_path)
        header_line = chapter_header_line(chapter_text)
        source_file = chapter_path.name

        def log(obj: dict) -> None:
            obj = {"ts": datetime.now(timezone.utc).isoformat(), "file": source_file, **obj}
            if run_log_path and not self.dry_run:
                run_log_path.parent.mkdir(parents=True, exist_ok=True)
                with run_log_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # All extract passes use the full raw file (including Chapter title line) for context.
        self.run_glossary_pass(source_file, chapter_text, log)
        self.run_timeline_pass(source_file, header_line, chapter_text, log)
        self.run_facts_pass(source_file, header_line, chapter_text, log)
        self.run_relations_pass(source_file, header_line, chapter_text, log)
        self.run_scene_pass(source_file, header_line, chapter_text, log)
        out_text = self.run_translate_pass(source_file, header_line, chapter_text, log)

        out_path = out_dir / chapter_path.name
        if not self.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(out_text, encoding="utf-8", newline="\n")
        log({"stage": "done", "out": str(out_path)})


def discover_chapters(chapters_dir: Path) -> list[Path]:
    return sorted(chapters_dir.glob("C*.txt"))


def resolve_prompt_file(root: Path, prompts_dir: Path, rel_or_name: str) -> Path:
    """Paths with a directory separator are resolved under project root; bare names use prompts_dir."""
    p = Path(rel_or_name)
    if p.is_absolute():
        return p
    s = str(rel_or_name).replace("\\", "/")
    if "/" in s:
        return (root / p).resolve()
    return (prompts_dir / p.name).resolve()


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def main() -> int:
    _try_load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=str, default="", help="Path to one chapter .txt")
    parser.add_argument("--all", action="store_true", help="Process all chapters/C*.txt")
    parser.add_argument("--chapters-dir", type=str, default="chapters")
    parser.add_argument("--out-dir", type=str, default=str(Path("translations") / "vi"))
    parser.add_argument("--prompts-dir", type=str, default="prompts")
    parser.add_argument("--glossary", type=str, default=str(Path("data") / "glossary.jsonl"))
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--resume", action="store_true", help="Skip if output file already exists")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "--validate-retry-whole-chunk",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("VALIDATE_RETRY_WHOLE_CHUNK", True),
        help="If legacy validate fails to return PASS/FIXED_VI, run a second full-chunk translate + validate (expensive). "
        "Use --no-validate-retry-whole-chunk to skip and keep first VI (then e.g. CJK line-fix only).",
    )
    parser.add_argument(
        "--no-translate-header",
        action="store_true",
        help="Keep the first line (Chapter N - …) in Chinese; translate the rest only (no separate title LLM call)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--context-tokens", type=int, default=int(os.environ.get("LLM_CONTEXT_TOKENS", "13000")))
    parser.add_argument("--completion-translate", type=int, default=int(os.environ.get("LLM_COMPLETION_TRANSLATE", "4500")))
    parser.add_argument(
        "--completion-translate-title",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_TRANSLATE_TITLE", "256")),
        help="Max tokens for test harness translate_title prompt only (main pipeline translates title inside normal chunks)",
    )
    parser.add_argument("--completion-extract", type=int, default=int(os.environ.get("LLM_COMPLETION_EXTRACT", "1800")))
    parser.add_argument("--completion-validate", type=int, default=int(os.environ.get("LLM_COMPLETION_VALIDATE", "3200")))
    parser.add_argument(
        "--completion-cjk-fix",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_CJK_FIX", "1536")),
        help="Max tokens for JSONL line-fix pass (translate-only CJK leak retry)",
    )
    parser.add_argument(
        "--translate-prompt",
        type=str,
        default=os.environ.get("TRANSLATE_PROMPT_FILE", "translate.txt"),
        help="Translate prompt file (name under prompts/ or path under project root)",
    )
    parser.add_argument(
        "--cjk-retry-prompt",
        type=str,
        default=os.environ.get("CJK_RETRY_PROMPT_FILE", "translate_retry_cjk_leak.txt"),
        help="CJK line-fix retry prompt (JSONL output), under prompts/ or project path",
    )
    parser.add_argument(
        "--translate-cjk-validate",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TRANSLATE_CJK_VALIDATE", False),
        help="After each translate chunk, fix lines that still contain CJK (translate stage only). Enable for Qwen/uncensored models.",
    )
    parser.add_argument(
        "--cjk-validate-retries",
        type=int,
        default=int(os.environ.get("CJK_VALIDATE_RETRIES", "3")),
        help="Max LLM calls per chunk for CJK line-fix (not whole-chunk retranslate)",
    )
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable event timeline extraction after glossary extraction",
    )
    parser.add_argument("--timeline-path", type=str, default="timeline/timeline_events.jsonl")
    parser.add_argument(
        "--timeline-max-chars",
        type=int,
        default=int(os.environ.get("LLM_TIMELINE_MAX_CHARS", os.environ.get("TIMELINE_MAX_CHARS", "9000"))),
        help="Upper bound on ZH chars per timeline chunk (after context budget)",
    )
    parser.add_argument("--completion-timeline", type=int, default=int(os.environ.get("LLM_COMPLETION_TIMELINE", "2500")))
    parser.add_argument(
        "--completion-facts",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_FACTS", "2300")),
        help="Max tokens for extract_entity_facts (metadata JSONL)",
    )
    parser.add_argument(
        "--completion-relations",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_RELATIONS", "2300")),
        help="Max tokens for extract_relation_edges (metadata JSONL)",
    )
    parser.add_argument(
        "--completion-scenes",
        type=int,
        default=int(os.environ.get("LLM_COMPLETION_SCENES", "2300")),
        help="Max tokens for extract_scene_segments (metadata JSONL)",
    )
    parser.add_argument("--glossary-inject-tokens", type=int, default=int(os.environ.get("GLOSSARY_INJECT_TOKENS", "1200")))
    parser.add_argument("--glossary-max-entries", type=int, default=int(os.environ.get("GLOSSARY_MAX_ENTRIES", "40")))
    parser.add_argument(
        "--safety-tokens",
        type=int,
        default=int(os.environ.get("LLM_SAFETY_TOKENS", "400")),
        help="Reserved tokens subtracted from context when sizing chunks",
    )
    parser.add_argument(
        "--existing-glossary-cap-tokens",
        type=int,
        default=int(os.environ.get("LLM_EXISTING_GLOSSARY_CAP_TOKENS", "900")),
        help="Max tokens for merged existing-glossary block in glossary extract",
    )
    parser.add_argument(
        "--title-glossary-inject-cap",
        type=int,
        default=int(os.environ.get("LLM_TITLE_GLOSSARY_INJECT_CAP", "800")),
        help="Cap (with glossary_inject_tokens) for title-line glossary selection",
    )
    parser.add_argument(
        "--title-glossary-max-entries",
        type=int,
        default=int(os.environ.get("LLM_TITLE_GLOSSARY_MAX_ENTRIES", "20")),
        help="Max glossary rows injected for title translation",
    )
    parser.add_argument(
        "--glossary-merge-max-entries",
        type=int,
        default=int(os.environ.get("LLM_GLOSSARY_MERGE_MAX_ENTRIES", "25")),
        help="Max rows when building cross-chunk existing glossary for extract",
    )
    parser.add_argument(
        "--glossary-timeline-max-entries",
        type=int,
        default=int(os.environ.get("LLM_GLOSSARY_TIMELINE_MAX_ENTRIES", "40")),
        help="Max chapter glossary rows for timeline pass",
    )
    parser.add_argument(
        "--glossary-metadata-max-entries",
        type=int,
        default=int(os.environ.get("LLM_GLOSSARY_METADATA_MAX_ENTRIES", "45")),
        help="Max chapter glossary rows for facts/relations/scenes",
    )
    parser.add_argument(
        "--metadata-facts-chunk-max-chars",
        type=int,
        default=int(os.environ.get("LLM_METADATA_FACTS_CHUNK_MAX_CHARS", "6000")),
        help="Upper bound on ZH chars per entity_facts chunk",
    )
    parser.add_argument(
        "--metadata-relations-chunk-max-chars",
        type=int,
        default=int(os.environ.get("LLM_METADATA_RELATIONS_CHUNK_MAX_CHARS", "7000")),
        help="Upper bound on ZH chars per relation_edges chunk",
    )
    parser.add_argument(
        "--metadata-scenes-chunk-max-chars",
        type=int,
        default=int(os.environ.get("LLM_METADATA_SCENES_CHUNK_MAX_CHARS", "7000")),
        help="Upper bound on ZH chars per scenes chunk",
    )

    parser.add_argument(
        "--facts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable entity facts extraction after timeline",
    )
    parser.add_argument(
        "--relations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable relation edge extraction after facts",
    )
    parser.add_argument(
        "--scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable scene/segment extraction after relations",
    )

    parser.add_argument("--metadata-path-entity-facts", type=str, default="metadata/entity_facts.jsonl")
    parser.add_argument("--metadata-path-relation-edges", type=str, default="metadata/relation_edges.jsonl")
    parser.add_argument("--metadata-path-scenes", type=str, default="metadata/scenes.jsonl")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    chapters_dir = (root / args.chapters_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    prompts_dir = (root / args.prompts_dir).resolve()
    glossary_path = (root / args.glossary).resolve()
    runs_dir = (root / args.runs_dir).resolve()

    client = client_from_env()
    budget = BudgetConfig(
        context_tokens=args.context_tokens,
        safety_tokens=args.safety_tokens,
        completion_translate=args.completion_translate,
        completion_translate_title=args.completion_translate_title,
        completion_extract=args.completion_extract,
        completion_validate=args.completion_validate,
        completion_timeline=args.completion_timeline,
        completion_facts=args.completion_facts,
        completion_relations=args.completion_relations,
        completion_scenes=args.completion_scenes,
        completion_cjk_fix=args.completion_cjk_fix,
        glossary_inject_tokens=args.glossary_inject_tokens,
        glossary_max_entries=args.glossary_max_entries,
        existing_glossary_cap_tokens=args.existing_glossary_cap_tokens,
        title_glossary_inject_cap=args.title_glossary_inject_cap,
        title_glossary_max_entries=args.title_glossary_max_entries,
        glossary_merge_max_entries=args.glossary_merge_max_entries,
        glossary_timeline_max_entries=args.glossary_timeline_max_entries,
        glossary_metadata_max_entries=args.glossary_metadata_max_entries,
        metadata_facts_chunk_max_chars=args.metadata_facts_chunk_max_chars,
        metadata_relations_chunk_max_chars=args.metadata_relations_chunk_max_chars,
        metadata_scenes_chunk_max_chars=args.metadata_scenes_chunk_max_chars,
    )

    translate_prompt_path = resolve_prompt_file(root, prompts_dir, args.translate_prompt)
    cjk_retry_prompt_path = resolve_prompt_file(root, prompts_dir, args.cjk_retry_prompt)

    pipe = Pipeline(
        root=root,
        client=client,
        budget=budget,
        prompts_dir=prompts_dir,
        glossary_path=glossary_path,
        dry_run=args.dry_run,
        validate_enabled=not args.no_validate,
        validate_retry_whole_chunk=args.validate_retry_whole_chunk,
        translate_header=not args.no_translate_header,
        timeline_enabled=args.timeline,
        timeline_path=(root / args.timeline_path).resolve() if args.timeline_path else (root / "timeline" / "timeline_events.jsonl"),
        timeline_max_chars=args.timeline_max_chars,
        facts_enabled=args.facts,
        entity_facts_path=(root / args.metadata_path_entity_facts).resolve(),
        relations_enabled=args.relations,
        relation_edges_path=(root / args.metadata_path_relation_edges).resolve(),
        scenes_enabled=args.scenes,
        scenes_path=(root / args.metadata_path_scenes).resolve(),
        translate_prompt_path=translate_prompt_path,
        translate_cjk_validate_enabled=args.translate_cjk_validate,
        cjk_validate_retries=args.cjk_validate_retries,
        cjk_retry_prompt_path=cjk_retry_prompt_path,
    )

    if args.chapter:
        ch = Path(args.chapter)
        paths = [ch.resolve() if ch.is_absolute() else (root / ch).resolve()]
    elif args.all:
        paths = discover_chapters(chapters_dir)
    else:
        print("Specify --chapter <file> or --all", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in paths:
        if not p.is_file():
            print(f"skip missing: {p}", file=sys.stderr)
            continue
        out_path = out_dir / p.name
        if args.resume and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"resume skip: {out_path}")
            continue
        run_log = runs_dir / f"{run_id}_{p.stem}.jsonl"
        print(f"process: {p.name}")
        try:
            pipe.process_file(p, out_dir, run_log_path=run_log)
        except Exception as e:
            print(f"ERROR {p.name}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
