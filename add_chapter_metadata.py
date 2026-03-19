#!/usr/bin/env python3
"""
Prepend metadata line to each chapter .txt file.

Filename pattern: C{number}-{title}.txt
Example output header:
  Chapter 135 - 正文 第一百三十五章 淫虐學院 | SourceFile: C135-正文 第一百三十五章 淫虐學院.txt

Use --dry-run to preview changes without writing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FILENAME_RE = re.compile(r"^C(\d+)-(.+)\.txt$", re.IGNORECASE)


def build_header(chapter_num: int, title: str, source_file: str) -> str:
    # Keep it in ONE line so downstream header translation keeps source file mapping.
    return f"Chapter {chapter_num} - {title.strip()} | SourceFile: {source_file}"


def strip_existing_metadata(text: str) -> str:
    """Remove leading 'Chapter N - ...' block and following blank lines."""
    if not text:
        return text
    # Handle UTF-8 BOM
    had_bom = text.startswith("\ufeff")
    if had_bom:
        text = text[1:]

    lines = text.splitlines(keepends=True)
    if not lines:
        return "\ufeff" + text if had_bom else text

    first = lines[0].lstrip("\ufeff")
    m = re.match(r"^Chapter\s+\d+\s+-\s+.+$", first.rstrip("\r\n"))
    if not m:
        return ("\ufeff" if had_bom else "") + text

    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    rest = "".join(lines[i:])
    return ("\ufeff" if had_bom else "") + rest


def process_file(path: Path, dry_run: bool, force: bool) -> tuple[str, str]:
    """
    Returns (status, message) where status is 'skip' | 'ok' | 'error'.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return "skip", f"skip (name pattern): {path.name}"

    chapter_num = int(m.group(1), 10)
    title_from_name = m.group(2).strip()
    expected_header = build_header(chapter_num, title_from_name, path.name)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return "error", f"{path.name}: read failed: {e}"

    body = raw
    if force:
        body = strip_existing_metadata(raw)

    # First non-empty line
    first_nonempty = None
    for line in body.splitlines():
        if line.strip():
            first_nonempty = line.strip()
            break

    if first_nonempty == expected_header:
        return "skip", f"skip (already has header): {path.name}"

    if first_nonempty and first_nonempty.startswith("Chapter ") and not force:
        return "skip", f"skip (has other header, use --force): {path.name}"

    new_content = expected_header + "\n\n" + body.lstrip("\ufeff")
    if dry_run:
        return "ok", f"[dry-run] would update: {path.name}"

    try:
        path.write_text(new_content, encoding="utf-8", newline="\n")
    except OSError as e:
        return "error", f"{path.name}: write failed: {e}"

    return "ok", f"updated: {path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepend Chapter N - title to chapter txt files.")
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=Path("chapters"),
        help="Directory containing C###-*.txt files (default: chapters)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing 'Chapter N - ...' header at top if present",
    )
    args = parser.parse_args()

    d = args.chapters_dir
    if not d.is_dir():
        print(f"Error: not a directory: {d}", file=sys.stderr)
        return 1

    files = sorted(d.glob("C*.txt"))
    if not files:
        print(f"No C*.txt files in {d}")
        return 0

    counts = {"ok": 0, "skip": 0, "error": 0}
    for path in files:
        status, msg = process_file(path, dry_run=args.dry_run, force=args.force)
        counts[status] = counts.get(status, 0) + 1
        print(msg)

    print(f"\nDone: updated={counts.get('ok', 0)} skipped={counts.get('skip', 0)} errors={counts.get('error', 0)}")
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
