# Eval question spec (for question authors)

Corpus: `data/processed/chunks.jsonl` (one JSON object per line; fields include `chunk_id`, `product_id`, `product`, `section`, `text`).
These are cleaned Wikipedia article chunks for 28 enterprise infrastructure products.

Write JSONL, one object per line, with exactly these fields:

```json
{"qid": "sd-a-01", "type": "single_doc", "phrasing": "paraphrase",
 "question": "...",
 "gold_answer": "one-sentence reference answer",
 "answer_keywords": [["apt", "advanced package tool"], ["dpkg"]],
 "evidence": [{"product_id": "ubuntu", "quote": "verbatim substring copied from a chunk's text"}],
 "notes": ""}
```

Rules
- `type`: `single_doc` (answer is in ONE chunk/section of one product), `multi_doc` (needs evidence from 2+ chunks, usually 2 different products, e.g. compare/contrast or "which of X and Y"), `unanswerable` (the corpus does NOT contain the answer).
- `evidence[].quote`: copied **verbatim** from the `text` field (exact characters, no ellipses), 8–30 words, and it must contain the fact that answers the question. A validator does exact substring matching after whitespace normalisation, so copy-paste carefully. One evidence item per required fact/chunk. For `unanswerable`, `evidence` is `[]`.
- `answer_keywords`: list of groups; a correct answer must contain at least one alternative from EVERY group (case-insensitive substring). Keep groups minimal (1–3 groups) and alternatives generous (numbers both as "2014" etc., common abbreviations). For `unanswerable`, use `[]`.
- `phrasing`: `lexical` if the question reuses distinctive words from the evidence; `paraphrase` if it asks the way a real IT admin / business user would, with different wording (synonyms, no copying rare terms). Aim for ~60% paraphrase.
- Questions must be answerable from the evidence text alone, unambiguous, with a short factual answer (a name, number, year, yes/no + reason, a short list). No questions whose answer is "it depends".
- Do NOT write questions about release dates / end-of-life / support end dates / latest version numbers of specific release cycles; those belong to a separate structured-data set.
- Most questions should name the product (as users do), but ~15% should not name it explicitly (e.g. "Which Linux distribution's name comes from a Nguni philosophy?").
- Prefer facts that matter to an enterprise IT buyer/operator: licensing, governance/vendor, architecture, supported protocols/platforms, compatibility, feature behaviour, history that explains current state.
- Before finalising each `unanswerable` question, grep the whole corpus to make sure the answer is really absent. Good unanswerable kinds: in-domain detail the articles don't contain (pricing, internal numbers, specific config defaults not mentioned), plausible-but-nonexistent features/versions, products not in the corpus. Avoid trick false premises that a careful reader could still "answer" by correcting the premise.
- Validate your file before finishing: every quote must be found verbatim in some chunk text whose product_id matches (`python3` one-liner is fine), every line must be valid JSON.
