"""Ablation: does prefixing each chunk with "[Product | Section]" before indexing help, and does
it make the metadata filter redundant? Runs the retrieval eval on a header-prefixed index;
compare with reports/retrieval_metrics.json (bare text, the default)."""
from ceeval.evaluate import run_retrieval

if __name__ == "__main__":
    print("== WITH contextual headers")
    run_retrieval(use_headers=True, out_name="retrieval_metrics_with_headers.json")
