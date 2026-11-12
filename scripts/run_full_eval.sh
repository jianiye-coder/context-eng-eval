#!/bin/sh
# Full eval: data quality gate -> retrieval metrics -> generation (5 configs) -> judge -> summary.
# Every step is resumable; re-run after an interruption and it continues where it stopped.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m ceeval.quality
$PY -m ceeval.evaluate retrieval
$PY -m ceeval.evaluate generate bm25 dense hybrid hybrid_meta routed
$PY -m ceeval.evaluate judge bm25 dense hybrid hybrid_meta routed
$PY -m ceeval.evaluate summarize
