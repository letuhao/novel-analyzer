import argparse
import os
import random
import re
import time
from urllib.parse import urljoin


def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Avoid collapsing paragraph breaks entirely: only collapse runs of spaces/tabs.
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip()
    return s


def _sanitize_filename(s: str) -> str:
    # Windows invalid filename characters: < > : " / \ | ? *
    s = s.replace("/", " ")
    s = s.replace("\\", " ")
    s = s.replace(":", " ")
    s = s.replace("：", " ")
    s = s.replace("*", " ")
    s = s.replace("?", " ")
    s = s.replace('"', " ")
    s = s.replace("<", " ")
    s = s.replace(">", " ")
    s = s.replace("|", " ")
    # Avoid control chars
    s = re.sub(r"[\x00-\x1f]", "", s)
    s = _normalize_ws(s)
    return s


def chinese_to_int(ch: str) -> int:
    """
    Convert simplified Chinese numerals in ranges like:
    - 十/十一/二十/二十一
    - 二百零一
    - 第一百七十一章 (after stripping 第/章)
    Supports up to 千 (<= 9999).
    """

    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {
        "十": 10,
        "百": 100,
        "千": 1000,
    }

    s = ch.strip()
    if not s:
        return 0

    # Fast path if it's already digits
    if s.isdigit():
        return int(s)

    total = 0
    num = 0
    for c in s:
        if c in digits:
            num = digits[c]
        elif c in units:
            unit = units[c]
            if num == 0:
                num = 1  # e.g. "十" => 10
            total += num * unit
            num = 0
        else:
            # Unknown char (e.g. punctuation). Ignore.
            pass
    total += num
    return total


def parse_chapter_number(chapter_title: str) -> int:
    # Example title: "正文 第二章、魔王的無奈"
    if "序章" in chapter_title or chapter_title.strip().startswith("正文 序章"):
        return 0

    m = re.search(r"第([零一二三四五六七八九十百千]+)章", chapter_title)
    if not m:
        # Fallback: try "第...章" without "正文"
        m2 = re.search(r"第(.+?)章", chapter_title)
        if m2:
            return chinese_to_int(m2.group(1))
        return -1

    return chinese_to_int(m.group(1))


def parse_chapter_title(page) -> str:
    # The chapter page typically has an H2 like: "正文 第二章、魔王的無奈"
    # Headless sometimes shows an age-gate overlay; we defensively add fallbacks.
    candidates = [("h2", "正文"), ("h1", "正文"), ("h2", None), ("h1", None)]

    for sel, must_contain in candidates:
        loc = page.locator(sel)
        if must_contain:
            loc = loc.filter(has_text=must_contain)
        loc = loc.first
        try:
            title = loc.text_content(timeout=5_000) or ""
        except Exception:
            title = ""
        title = _normalize_ws(title)
        if title:
            return title

    return ""


def extract_chapter_text(full_text: str) -> str:
    """
    Extract story text between the UI font-size marker ("繁 A- A+")
    and the navigation/footers ("上一章"/"目錄").
    """

    text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    # The site shows font controls as either "繁 A- A+" or "简 A- A+"
    # depending on mode/headless rendering, so accept both.
    start_re = re.compile(r"(繁|简)\s*A-\s*A\+")
    m = start_re.search(text)
    start = m.end() if m else 0

    end_keys = ["上一章", "目錄", "下一章"]
    end = len(text)
    for k in end_keys:
        i = text.find(k, start)
        if i != -1:
            end = min(end, i)

    chapter_text = text[start:end].strip()

    # Drop trailing page separators like "____"
    chapter_text = re.sub(r"\n?_{2,}\s*$", "", chapter_text)

    # Keep paragraph spacing reasonable
    chapter_text = re.sub(r"\n{3,}", "\n\n", chapter_text)
    return chapter_text


def find_next_url(page, current_url: str) -> str | None:
    # Prefer the anchor with visible text "下一章".
    try:
        next_link = page.locator("a", has_text=re.compile(r"\s*下一章\s*")).first
        href = next_link.get_attribute("href")
        if href:
            return urljoin(current_url, href)
    except Exception:
        pass

    # Fallback: iterate all anchors and pick the first one whose rendered text
    # contains "下一章" and has a valid href.
    anchors = page.locator("a")
    candidate: str | None = None
    for i in range(anchors.count()):
        a = anchors.nth(i)
        try:
            href = a.get_attribute("href")
            if not href:
                continue
            txt = _normalize_ws(a.inner_text() or "")
            if "下一章" not in txt:
                continue
            # Prefer stable chapter URL pattern.
            if "/book/732/" in href:
                candidate = href
                break
            if candidate is None:
                candidate = href
        except Exception:
            continue

    if not candidate:
        return None
    return urljoin(current_url, candidate)


def fetch_one_chapter(context, url: str, output_dir: str) -> str | None:
    page = context.new_page()
    page.set_default_timeout(30_000)
    page.set_default_navigation_timeout(30_000)

    # Basic human-like delay before navigation.
    time.sleep(random.uniform(0.6, 1.2))
    page.goto(url, wait_until="domcontentloaded")

    # If there's an age-gate modal, click "我已滿 18 歲" once visible.
    try:
        age_btn = page.locator("text=我已滿 18 歲").first
        if age_btn.count() > 0 and age_btn.is_visible():
            age_btn.click()
            page.wait_for_timeout(800)
    except Exception:
        pass

    # Wait for the title area to exist.
    try:
        page.locator("h2, h1").first.wait_for(timeout=10_000)
    except Exception:
        pass

    title = parse_chapter_title(page)
    body_text = page.locator("body").inner_text()
    chapter_text = extract_chapter_text(body_text)

    if not title or not chapter_text:
        page.close()
        return None

    num = parse_chapter_number(title)
    if num < 0:
        # Keep a stable ordering even if parsing failed.
        # (Try to extract from URL: /book/732/155785/)
        pass

    prefix = "C000" if num == 0 else f"C{num:03d}" if num >= 1 else "CXXX"
    safe_title = _sanitize_filename(title)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}-{safe_title}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(chapter_text)

    # Mimic reading time.
    time.sleep(random.uniform(0.6, 1.5))

    next_url = find_next_url(page, url)
    page.close()
    return next_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", default="https://m.hotupub.net/book/732/0/")
    parser.add_argument("--output-dir", default="chapters")
    parser.add_argument("--max-chapters", type=int, default=9999)
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--delay-min", type=float, default=0.6)
    parser.add_argument("--delay-max", type=float, default=1.5)
    args = parser.parse_args()

    # Import inside to keep error message clearer.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: playwright. Install with:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
        ) from e

    global random
    random_delay_min = args.delay_min
    random_delay_max = args.delay_max

    def _rand_delay(a, b):
        return random.uniform(a, b)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)

        # Use a typical UA; Playwright sets some defaults but UA helps.
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            # This site appears to have an invalid/odd certificate on some paths.
            # We ignore TLS errors to allow scraping to proceed.
            ignore_https_errors=True,
            viewport={"width": 900, "height": 1200},
        )

        url = args.start_url
        visited = set()

        for idx in range(args.max_chapters):
            if url in visited:
                print(f"[stop] loop detected at {url}")
                break
            visited.add(url)

            print(f"[{idx+1}] Fetch: {url}")
            context_sleep = _rand_delay(random_delay_min, random_delay_max)
            time.sleep(context_sleep)

            next_url = fetch_one_chapter(context, url, args.output_dir)
            if not next_url:
                print("[stop] no next chapter link found")
                break

            url = next_url

        context.close()
        browser.close()


if __name__ == "__main__":
    main()

