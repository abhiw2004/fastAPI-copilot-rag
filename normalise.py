import json
from pathlib import Path

from bs4 import BeautifulSoup

CORPUS_DIR  = Path("corpus")
CLEAN_DIR   = Path("corpus_clean")
REPORT_PATH = Path("cleaning_report.json")

NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside"]

NOISE_CLASS_HINTS = [
    "nav", "sidebar", "toc", "footer", "header", "breadcrumb",
    "banner", "sponsor", "skip-link", "announce", "search",
]

CONTENT_SELECTORS = [
    "article",
    "main",
    "[role=main]",
    ".md-content__inner",
    "#content",
    ".content",
]

MIN_CONTENT_CHARS = 200


def strip_noise(soup: BeautifulSoup) -> None:
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            if not tag.decomposed:
                tag.decompose()

    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        classes  = " ".join(tag.get("class") or []).lower()
        tag_id   = (tag.get("id") or "").lower()
        if any(hint in f"{classes} {tag_id}" for hint in NOISE_CLASS_HINTS):
            tag.decompose()


def find_main_content(soup: BeautifulSoup):
    for selector in CONTENT_SELECTORS:
        match = soup.select_one(selector)
        if match and len(match.get_text(strip=True)) >= MIN_CONTENT_CHARS:
            return match, selector
    return soup.body or soup, "fallback_body"


def clean_html(raw_html: str) -> tuple[str, dict]:
    soup         = BeautifulSoup(raw_html, "html.parser")
    raw_text_len = len(soup.get_text(strip=True))

    strip_noise(soup)
    content, selector_used = find_main_content(soup)

    clean_text_len = len(content.get_text(strip=True))
    stats = {
        "selector_used":  selector_used,
        "raw_char_count": raw_text_len,
        "clean_char_count": clean_text_len,
        "kept_ratio": round(clean_text_len / raw_text_len, 3) if raw_text_len else 0.0,
    }
    return str(content), stats


def main():
    metadata = json.loads((CORPUS_DIR / "metadata.json").read_text(encoding="utf-8"))
    report   = {}

    for rel_path in metadata:
        raw_path = CORPUS_DIR / rel_path
        if not raw_path.exists():
            print(f"  MISSING: {rel_path}")
            continue

        cleaned_html, stats = clean_html(raw_path.read_text(encoding="utf-8"))

        clean_path = CLEAN_DIR / rel_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text(cleaned_html, encoding="utf-8")

        report[rel_path] = stats

        flag = ""
        if stats["raw_char_count"] > 0 and stats["kept_ratio"] < 0.05:
            flag = "  <-- low kept ratio"
        elif stats["selector_used"] == "fallback_body" and stats["raw_char_count"] > 3000:
            flag = "  <-- fallback on large page"

        print(
            f"{rel_path}: {stats['selector_used']}, "
            f"kept {stats['clean_char_count']}/{stats['raw_char_count']} chars{flag}"
        )

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone. Cleaned files in {CLEAN_DIR}/, report in {REPORT_PATH}")


if __name__ == "__main__":
    main()
