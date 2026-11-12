# Data card: enterprise infrastructure knowledge base (v0.1, snapshot 2026-11-11)

Two AI-ready data products built from public sources for 28 enterprise infrastructure products
(operating systems, hypervisors, container platforms, databases, storage, data platforms: the
software stack that runs on data-center servers).

## 1. `data/processed/chunks.jsonl`: unstructured knowledge (656 chunks)

| | |
|---|---|
| Source | English Wikipedia article per product, plain-text extract via the MediaWiki API |
| Pinning | Each doc stores its `revid`, so every chunk links to the exact revision (`source_url` has `oldid=`) |
| Size | 656 chunks, 83,360 words; 25–234 words per chunk (mean 127) |
| Coverage | os 211 · database 209 · storage 101 · containers 67 · virtualization 43 · data-platform 25 chunks |
| Revision dates | 2026-07-09 to 2026-11-11 |
| License | CC BY-SA 4.0 (Wikipedia contributors). Derived chunks keep that license. |

**Schema** (validated with pydantic in `ceeval/quality.py`):

| field | meaning |
|---|---|
| `chunk_id` | `{product_id}::{section#}-{section-slug}::{chunk#}`, stable as long as the revision is unchanged |
| `product_id`, `product`, `category` | product metadata (slug from endoflife.date, display name, stack layer) |
| `doc_title`, `section` | article title and heading path, e.g. `History > Early years` |
| `text`, `n_words` | cleaned chunk body |
| `source_url`, `revid`, `doc_date`, `license` | provenance |
| `content_hash` | sha1 prefix of the text, used for exact-duplicate detection |

**Processing**
1. Split on wiki headings into a section tree and drop boilerplate sections (References, See also, External links, ...).
2. Clean: strip pronunciation residue left by the extract (e.g. `Ubuntu (  uu-BUUN-too)`), LaTeX residue, and stray whitespace.
3. Fold sections under 25 words (often headings whose content was a table) into the next section.
4. Pack whole sentences greedily to about 180 words (hard max 260), with a one-sentence overlap between neighbours.

## 2. `data/processed/lifecycle.db`: structured lifecycle data (511 release rows)

| | |
|---|---|
| Source | endoflife.date API v1 (data MIT-licensed) |
| Tables | `products(product_id, product, category)`, `releases(product_id, cycle, codename, release_date, is_lts, eoas_date, eol_date, eoes_date, latest_version, latest_date)`, `meta(as_of)` |
| Coverage | 511 release cycles; release dates 1995-06-13 to 2026-10-30 |
| As-of | `2026-11-11`. "Still supported" is always evaluated against this literal date, never `date('now')`, so answers are reproducible |

Column semantics are documented in `ceeval/sql_tool.py:SCHEMA_DOC`. That same text is what the
text-to-SQL model sees, so documentation and behaviour cannot drift apart.
`eoas`/`eol`/`eoes` map vendor-specific phase names (full / maintenance / extended support, and
so on) onto three common phases. The vendor's own label for each phase is kept in the raw snapshot under `labels`.

## Quality checks (`python -m ceeval.quality` → `reports/data_quality.json`, exit 1 on failure)

| check | blocking | current |
|---|:-:|---:|
| chunk schema validation (pydantic) | yes | 0 errors |
| duplicate chunk ids | yes | 0 |
| exact duplicate text (content hash) | yes | 0 |
| near duplicates (5-gram shingle Jaccard ≥ 0.8) | report | 0 |
| chunks under 25 words | yes | 0 |
| markup residue (`{{`, `[[`, `<ref`, `{\displaystyle`) | yes | 0 |
| garbled text (symbol ratio > 8%) | report | 0 |
| every catalog product has chunks | yes | 28/28 |
| lifecycle: column schema matches contract | yes | ok |
| lifecycle: dates parse as ISO | yes | 0 bad |
| lifecycle: eol_date before release_date | yes | 0 |
| lifecycle: orphan product FK | yes | 0 |
| lifecycle: missing eol_date | report | 20 (rolling or unannounced EOL, expected) |

The first ingestion pass failed two of these checks: 32 chunk-ID collisions (the same heading
text in different parts of an article) and 28 chunks under 25 words. Both were fixed in
`ingest.py` (section-index IDs, small-section folding). Both checks are now blocking and run in CI.

## Known limitations
- Wikipedia is secondary and sometimes stale. For production support answers the source of truth
  would be vendor docs, where licensing allows it.
- Tables in the articles (release histories, feature matrices) are lost by the plain-text
  extract. That is by design here, since lifecycle facts come from the structured table, but other table facts are not covered.
- endoflife.date maps each vendor's own lifecycle phases onto three columns. Some products
  have no `eoas` (active support) phase.
- Coverage is uneven: ONTAP's article yields 83 chunks, vSphere's only 2.

## Refresh
`python -m ceeval.fetch` (only fetches missing files; delete `data/raw/*` to refresh), then
`python -m ceeval.ingest && python -m ceeval.quality`. Changing a revision changes chunk text, so
gold evidence in the eval set is stored as **verbatim quotes** rather than chunk ids, and is
re-resolved to chunk ids at eval time. `eval/build_eval_set.py` fails loudly if a quote stops resolving.
