import json
import re
import time
from datetime import date
from pathlib import Path

import markdown
import requests

OUTPUT_DIR    = Path("corpus")
METADATA_PATH = OUTPUT_DIR / "metadata.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; KnowledgeCopilotBot/1.0)"}

BASE  = "https://fastapi.tiangolo.com"
TODAY = "2024-01-01"

PAGES = [
    (f"{BASE}/tutorial/first-steps/",                    "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/path-params/",                    "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/query-params/",                   "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/body/",                           "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/query-params-str-validations/",   "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/body-multiple-params/",           "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/header-params/",                  "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/response-model/",                 "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/handling-errors/",                "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/dependencies/",                   "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/dependencies/sub-dependencies/",  "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/security/first-steps/",           "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/security/get-current-user/",      "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/security/oauth2-jwt/",            "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/middleware/",                     "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/cors/",                           "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/sql-databases/",                  "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/bigger-applications/",            "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/background-tasks/",               "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/testing/",                        "tutorial",      TODAY, "public"),
    (f"{BASE}/tutorial/debugging/",                      "tutorial",      TODAY, "public"),
    (f"{BASE}/advanced/middleware/",                     "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/security/oauth2-scopes/",         "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/security/http-basic-auth/",       "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/custom-response/",                "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/additional-responses/",           "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/websockets/",                     "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/settings/",                       "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/testing-dependencies/",           "advanced",      TODAY, "public"),
    (f"{BASE}/advanced/behind-a-proxy/",                 "advanced",      TODAY, "public"),
    (f"{BASE}/deployment/concepts/",                     "deployment",    TODAY, "public"),
    (f"{BASE}/deployment/docker/",                       "deployment",    TODAY, "public"),
    (f"{BASE}/deployment/https/",                        "deployment",    TODAY, "public"),
    (f"{BASE}/how-to/general/",                          "how_to",        TODAY, "public"),
    (f"{BASE}/how-to/graphql/",                          "how_to",        TODAY, "public"),
    (f"{BASE}/reference/fastapi/",                       "reference",     TODAY, "public"),
    (f"{BASE}/reference/parameters/",                    "reference",     TODAY, "public"),
    (f"{BASE}/reference/security/",                      "reference",     TODAY, "public"),
    (f"{BASE}/release-notes/",                           "release_notes", TODAY, "public"),
]

# Versioned snapshots used to test stale-doc handling
OLD_PAGES = [
    ("0.68.0", "advanced/security/oauth2-scopes.md", "outdated", "2021-07-01", "public"),
    ("0.68.0", "advanced/middleware.md",              "outdated", "2021-07-01", "public"),
    ("0.68.0", "tutorial/sql-databases.md",           "outdated", "2021-07-01", "public"),
    ("0.68.0", "tutorial/background-tasks.md",        "outdated", "2021-07-01", "public"),
    ("0.68.0", "tutorial/security/oauth2-jwt.md",     "outdated", "2021-07-01", "public"),
]
OLD_PAGE_BASE = "https://raw.githubusercontent.com/tiangolo/fastapi/{tag}/docs/en/docs/{path}"


def slugify(url: str) -> str:
    path = url.rstrip("/").split("//", 1)[-1].split("/", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.strip("/"))
    return slug or "index"


def fetch_page(url: str, retries: int = 3, delay: float = 1.5) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            print(f"  attempt {attempt} failed for {url}: {exc}")
            time.sleep(delay)
    return None


def scrape(pages, metadata):
    for url, doc_type, last_updated, access_level in pages:
        slug         = slugify(url)
        doc_type_dir = OUTPUT_DIR / doc_type
        doc_type_dir.mkdir(parents=True, exist_ok=True)
        filepath     = doc_type_dir / f"{slug}.html"

        print(f"Fetching {url} ...")
        html = fetch_page(url)
        if html is None:
            print(f"  SKIPPED: {url}")
            continue

        filepath.write_text(html, encoding="utf-8")
        print(f"  saved -> {filepath}")

        metadata[str(filepath.relative_to(OUTPUT_DIR))] = {
            "source":       url,
            "doc_type":     doc_type,
            "last_updated": last_updated,
            "access_level": access_level,
            "scraped_on":   date.today().isoformat(),
        }
        time.sleep(1.0)


def scrape_old_pages(pages, metadata):
    for tag, doc_path, doc_type, last_updated, access_level in pages:
        url          = OLD_PAGE_BASE.format(tag=tag, path=doc_path)
        slug         = slugify(doc_path) + f"_{tag}"
        doc_type_dir = OUTPUT_DIR / doc_type
        doc_type_dir.mkdir(parents=True, exist_ok=True)
        filepath     = doc_type_dir / f"{slug}.html"

        print(f"Fetching {url} ...")
        md_text = fetch_page(url)
        if md_text is None:
            print(f"  SKIPPED: {url}")
            continue

        html = markdown.markdown(md_text, extensions=["extra"])
        filepath.write_text(html, encoding="utf-8")
        print(f"  saved -> {filepath}")

        metadata[str(filepath.relative_to(OUTPUT_DIR))] = {
            "source":       url,
            "doc_type":     doc_type,
            "last_updated": last_updated,
            "access_level": access_level,
            "scraped_on":   date.today().isoformat(),
        }
        time.sleep(1.0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

    scrape(PAGES, metadata)
    scrape_old_pages(OLD_PAGES, metadata)

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nDone. {len(metadata)} pages recorded in {METADATA_PATH}")


if __name__ == "__main__":
    main()
