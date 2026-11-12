"""Retrieval variants over the chunk data product.

bm25         lexical (Okapi BM25) over chunk text
dense        bge-small-en-v1.5 embeddings, cosine similarity
hybrid       Reciprocal Rank Fusion (RRF) of bm25 + dense
hybrid_meta_hard  hybrid restricted to the products named in the question (hard metadata
                  filter), with a per-product quota so multi-product questions get evidence for each
hybrid_meta       same, but one of the k slots is reserved for the best *unfiltered* hit
                  ("escape hatch"), because a hard filter fails when the product a question
                  mentions is not the product it is about (see README: failure analysis)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from rank_bm25 import BM25Okapi

from .config import EMBED_MODEL, EMBED_QUERY_PREFIX, INDEX_DIR, RAW_EOL, RRF_K, TOP_K
from .ingest import Chunk, load_chunks

STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this
which what who whom when where why how does do did it its into than then there their they them can
could should would will may might about over under between both each any all such not no""".split())

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


# ---------------------------------------------------------------- product (metadata) detection
MANUAL_ALIASES = {
    "rhel": ["rhel", "red hat enterprise linux"],
    "ubuntu": ["ubuntu"],
    "debian": ["debian"],
    "sles": ["sles", "suse linux enterprise", "suse enterprise", "suse"],
    "windows-server": ["windows server"],
    "oracle-linux": ["oracle linux"],
    "rocky-linux": ["rocky linux", "rocky"],
    "almalinux": ["almalinux", "alma linux"],
    "centos": ["centos"],
    "esxi": ["esxi", "vmware esx", "esx"],
    "vcenter": ["vcenter", "vsphere"],
    "proxmox-ve": ["proxmox"],
    "kubernetes": ["kubernetes", "k8s"],
    "red-hat-openshift": ["openshift"],
    "docker-engine": ["docker"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mssqlserver": ["sql server", "mssql"],
    "oracle-database": ["oracle database", "oracle db", "oracle rdbms"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "mariadb": ["mariadb"],
    "netapp-ontap": ["ontap", "netapp"],
    "truenas": ["truenas", "freenas"],
    "ceph": ["ceph"],
    "apache-kafka": ["kafka"],
    "apache-spark": ["spark"],
    "elasticsearch": ["elasticsearch"],
}


def _alias_table() -> list[tuple[str, str]]:
    table: dict[str, str] = {}
    for pid, names in MANUAL_ALIASES.items():
        for n in names:
            table[n] = pid
        p = RAW_EOL / f"{pid}.json"
        if p.exists():  # endoflife.date ships curated aliases; reuse them as metadata
            for a in json.loads(p.read_text())["result"].get("aliases", []):
                if len(a) >= 4 and a.isalpha():
                    table.setdefault(a.lower(), pid)
    return sorted(table.items(), key=lambda kv: -len(kv[0]))  # longest match first


ALIASES = _alias_table()


def detect_products(question: str) -> list[str]:
    q = " " + re.sub(r"[^a-z0-9.]+", " ", question.lower()) + " "
    found: list[str] = []
    for alias, pid in ALIASES:
        pat = f" {alias} "
        if pat in q:
            q = q.replace(pat, " # ")  # consume, so "oracle linux" doesn't also fire "linux"
            if pid not in found:
                found.append(pid)
    return found


# ---------------------------------------------------------------- retrievers
@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, use_headers: bool = False):
        self.chunks = chunks if chunks is not None else load_chunks()
        # Index bare chunk text by default. The "[Product | Section]" header variant was measured
        # in scripts/ablation_headers.py and did not help (README: ablation); the LLM context
        # still carries product/section/source per chunk (pipeline.format_context).
        self.texts = [c.contextual_text() if use_headers else c.text for c in self.chunks]
        self.pid = np.array([c.product_id for c in self.chunks])

    # -- lexical
    @cached_property
    def bm25(self) -> BM25Okapi:
        return BM25Okapi([tokenize(t) for t in self.texts])

    def bm25_scores(self, q: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(q)))

    # -- dense
    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL, device="cpu")

    @cached_property
    def emb(self) -> np.ndarray:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1((EMBED_MODEL + "".join(self.texts)).encode()).hexdigest()[:12]
        path = INDEX_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        e = self.model.encode(self.texts, batch_size=8, normalize_embeddings=True,
                              show_progress_bar=False)
        np.save(path, e)
        return e

    def dense_scores(self, q: str) -> np.ndarray:
        qv = self.model.encode([EMBED_QUERY_PREFIX + q], normalize_embeddings=True)[0]
        return self.emb @ qv

    # -- fusion
    @staticmethod
    def _ranks(scores: np.ndarray, mask: np.ndarray, depth: int = 50) -> dict[int, int]:
        s = np.where(mask, scores, -np.inf)
        order = np.argsort(-s)[:depth]
        return {int(i): r for r, i in enumerate(order) if np.isfinite(s[i])}

    def search(self, q: str, variant: str, k: int = TOP_K) -> list[Hit]:
        mask = np.ones(len(self.chunks), dtype=bool)
        products: list[str] = []
        if variant in ("hybrid_meta", "hybrid_meta_hard"):
            products = detect_products(q)
            if products:
                mask = np.isin(self.pid, products)

        if variant == "bm25":
            s = self.bm25_scores(q)
        elif variant == "dense":
            s = self.dense_scores(q)
        elif variant in ("hybrid", "hybrid_meta", "hybrid_meta_hard"):
            bm, de = self.bm25_scores(q), self.dense_scores(q)
            s = self._rrf(bm, de, mask)
            if variant == "hybrid_meta" and products:
                unfiltered = self._rrf(bm, de, np.ones(len(self.chunks), dtype=bool))
        else:
            raise ValueError(variant)

        s = np.where(mask, s, -np.inf)
        order = [int(i) for i in np.argsort(-s) if np.isfinite(s[i])]
        if len(products) > 1:
            order = self._quota(order, products, k)
        if variant == "hybrid_meta" and products:
            best_global = int(np.argmax(unfiltered))
            if best_global not in order[:k]:
                order = order[: k - 1] + [best_global] + order[k - 1:]
                s[best_global] = unfiltered[best_global]
        return [Hit(self.chunks[i], float(s[i]), r) for r, i in enumerate(order[:k])]

    def _rrf(self, bm: np.ndarray, de: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rb, rd = self._ranks(bm, mask), self._ranks(de, mask)
        s = np.full(len(self.chunks), -np.inf)
        for i in set(rb) | set(rd):
            s[i] = sum(1.0 / (RRF_K + r[i] + 1) for r in (rb, rd) if i in r)
        return s

    def _quota(self, order: list[int], products: list[str], k: int) -> list[int]:
        """Guarantee each named product at least floor(k / n_products) of the top-k slots."""
        per = max(1, k // len(products))
        picked: list[int] = []
        for p in products:
            picked += [i for i in order if self.pid[i] == p][:per]
        picked_set = set(picked)
        rest = [i for i in order if i not in picked_set]
        # keep fused order among the guaranteed picks, then fill
        picked.sort(key=order.index)
        return picked + rest


VARIANTS = ["bm25", "dense", "hybrid", "hybrid_meta_hard", "hybrid_meta"]
