"""Fast unit tests (no LLM, no embedding model)."""
import json

from ceeval import quality
from ceeval.ingest import merge_small_sections, pack_sentences, split_sections
from ceeval.metrics import all_found_at_k, keyword_match, recall_at_k, reciprocal_rank
from ceeval.pipeline import is_abstention, route
from ceeval.retrieval import detect_products
from ceeval.sql_tool import extract_sql, validate


# ---- metrics
def test_keyword_match_word_boundaries():
    assert keyword_match("There are 3 supported versions.", [["3", "three"]])
    assert not keyword_match("There are 13 versions.", [["3"]])
    assert not keyword_match("It is not required.", [["no"]])
    assert keyword_match("skipped due to superstitions", [["superstition"]])
    assert not keyword_match("version 9.01", [["9.0"]])


def test_keyword_match_dates_in_any_common_format():
    groups = [["2032-05-31"]]
    assert keyword_match("Maintenance ends on May 31, 2032 [1].", groups)
    assert keyword_match("ends 31 May 2032", groups)
    assert not keyword_match("ends May 30, 2032", groups)


def test_keyword_match_requires_every_group():
    assert keyword_match("KVM and LXC", [["kvm"], ["lxc"]])
    assert not keyword_match("KVM only", [["kvm"], ["lxc"]])


def test_retrieval_metrics():
    gold = [{"a", "a2"}, {"b"}]
    ret = ["x", "a2", "y", "b"]
    assert recall_at_k(ret, gold, 2) == 0.5
    assert recall_at_k(ret, gold, 4) == 1.0
    assert all_found_at_k(ret, gold, 4) and not all_found_at_k(ret, gold, 3)
    assert reciprocal_rank(ret, gold) == 0.5
    assert reciprocal_rank(["x"], gold) == 0.0


# ---- ingestion
def test_split_sections_drops_boilerplate_and_nests_paths():
    text = "Lead para.\n\n== History ==\nOld.\n\n=== Early ===\nEarly text.\n\n== References ==\nref"
    secs = split_sections(text)
    assert [p for p, _ in secs] == [["Overview"], ["History"], ["History", "Early"]]


def test_small_sections_are_merged_forward():
    secs = [(["A"], "tiny"), (["B"], " ".join(["word"] * 40))]
    out = merge_small_sections(secs)
    assert len(out) == 1 and out[0][1].startswith("A: tiny")


def test_pack_sentences_never_exceeds_max_and_overlaps():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(100))
    chunks = pack_sentences(body)
    assert all(len(c.split()) <= 260 for c in chunks)
    assert chunks[0].split(". ")[-1] in chunks[1]  # one-sentence overlap


# ---- quality gate catches what it should
def test_quality_flags_duplicate_ids_and_short_chunks(tmp_path, monkeypatch):
    base = json.loads(quality.CHUNKS_PATH.read_text().splitlines()[0])
    bad = [base, dict(base), dict(base, chunk_id=base["chunk_id"][:-1] + "9", text="too short",
                                   n_words=2, content_hash="zz")]
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in bad))
    monkeypatch.setattr(quality, "CHUNKS_PATH", p)
    rep = quality.check_chunks()
    assert "duplicate_ids" in rep["blocking_failures"]
    assert "exact_duplicates" in rep["blocking_failures"]
    assert "too_short" in rep["blocking_failures"]


# ---- metadata, routing, SQL guardrails
def test_detect_products_longest_match():
    assert detect_products("How does Oracle Linux relate to RHEL?") == ["oracle-linux", "rhel"]
    assert detect_products("Is Postgres on BSD?") == ["postgresql"]
    assert detect_products("What is a hypervisor?") == []


def test_router():
    assert route("When does RHEL 9 reach end of life?") == "sql"
    assert route("What is the end-of-support date for NetApp ONTAP 9.14.1?") == "sql"
    assert route("Who develops RHEL?") == "text"
    assert route("When does support end?") == "text"  # no product -> nothing to look up


def test_sql_guardrails():
    assert validate("SELECT eol_date FROM releases WHERE product_id='rhel' AND cycle='9'", "RHEL 9") is None
    assert validate("DELETE FROM releases") is not None
    assert validate("SELECT 1 FROM releases; DROP TABLE releases") is not None
    assert validate("SELECT * FROM users") is not None
    assert validate("SELECT * FROM releases WHERE eol_date > date('now')") is not None
    # the model must not silently answer about a different version than the one asked
    assert validate("SELECT eol_date FROM releases WHERE cycle='20.04'", "When does Ubuntu 30.04 end?") is not None
    assert validate("SELECT eol_date FROM releases WHERE cycle='9.6'", "When does RHEL 9 end?") is not None
    assert validate("SELECT cycle FROM releases WHERE cycle LIKE '15.%'", "Which SLES 15 service packs?") is None
    assert validate("SELECT cycle FROM releases ORDER BY release_date DESC LIMIT 3", "the top 3 newest releases") is None
    assert validate("SELECT * FROM releases, sqlite_master") is not None
    assert validate("SELECT name FROM products, users") is not None
    assert validate("SELECT replace(cycle, '.', '') FROM releases") is None
    assert validate("SELECT COUNT(*) FROM (SELECT cycle FROM releases)") is None


def test_extract_sql_from_fenced_reply():
    assert extract_sql("```sql\nSELECT 1 FROM releases;\n```") == "SELECT 1 FROM releases"


def test_abstention_detection():
    assert is_abstention("I don't know based on the provided sources.")
    # refusals worded differently (found in review)
    for t in ["The sources don't mention Ubuntu 30.04.", "I do not know.",
              "The provided sources do not have that information.", "This isn't stated in the sources."]:
        assert is_abstention(t), t
    # a negative *fact* with a citation is an answer, not a refusal
    assert not is_abstention("Debian does not include non-free firmware by default [1].")
    assert not is_abstention("Proxmox VE uses the AGPLv3 [1]. I don't know based on the provided sources about pricing.")
    assert not is_abstention("Kubernetes was originally designed by Google; the sources do not mention "
                             "the exact team size, but the CNCF now maintains it [1]. " * 2)
