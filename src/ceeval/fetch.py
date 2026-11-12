"""Fetch a pinned snapshot of the raw sources.

Unstructured: Wikipedia plain-text extracts (CC BY-SA 4.0), revision id pinned.
Structured:   endoflife.date API v1 release cycles (MIT licensed data).

Run once; the snapshot is committed so the rest of the pipeline is offline/reproducible.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

from .config import PRODUCTS, RAW_EOL, RAW_WIKI, USER_AGENT

WIKI_API = "https://en.wikipedia.org/w/api.php"
EOL_API = "https://endoflife.date/api/v1/products/{slug}/"


def _get(url: str, params: dict | None = None, tries: int = 6) -> dict:
    for i in range(tries):
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
        # Respect rate limiting (Wikipedia returns 429 + Retry-After).
        retry_after = r.headers.get("retry-after", "0")
        time.sleep(max(2 ** i, int(retry_after) if retry_after.isdigit() else 0) + 1)
    raise RuntimeError(f"failed to fetch {url} {params}")


def fetch_wiki(slug: str, title: str) -> None:
    out = RAW_WIKI / f"{slug}.json"
    if out.exists():
        return
    data = _get(WIKI_API, {
        "action": "query", "prop": "extracts|revisions", "rvprop": "ids|timestamp",
        "explaintext": 1, "exsectionformat": "wiki", "redirects": 1,
        "titles": title, "format": "json",
    })
    page = next(iter(data["query"]["pages"].values()))
    rev = page["revisions"][0]
    record = {
        "product_id": slug,
        "title": page["title"],
        "url": f"https://en.wikipedia.org/w/index.php?title={page['title'].replace(' ', '_')}&oldid={rev['revid']}",
        "revid": rev["revid"],
        "rev_timestamp": rev["timestamp"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "CC BY-SA 4.0",
        "text": page["extract"],
    }
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))


def fetch_eol(slug: str) -> None:
    out = RAW_EOL / f"{slug}.json"
    if out.exists():
        return
    data = _get(EOL_API.format(slug=slug))
    data["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def main() -> None:
    RAW_WIKI.mkdir(parents=True, exist_ok=True)
    RAW_EOL.mkdir(parents=True, exist_ok=True)
    for slug, (_label, _cat, title) in PRODUCTS.items():
        fetch_wiki(slug, title)
        fetch_eol(slug)
        print(f"fetched {slug}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
