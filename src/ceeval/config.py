"""Central configuration: paths, the product catalog, and model/eval settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_WIKI = DATA / "raw" / "wiki"
RAW_EOL = DATA / "raw" / "eol"
PROCESSED = DATA / "processed"
CHUNKS_PATH = PROCESSED / "chunks.jsonl"
DB_PATH = PROCESSED / "lifecycle.db"
INDEX_DIR = PROCESSED / "index"
EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
REPORTS = ROOT / "reports"

# Snapshot date. All "is it still supported?" logic is evaluated against this date,
# never against date('now'), so results are reproducible.
AS_OF = "2026-11-11"

USER_AGENT = "context-eng-eval/0.1 (educational research project)"

# product_id (endoflife.date slug) -> (display label, category, Wikipedia title)
PRODUCTS: dict[str, tuple[str, str, str]] = {
    "rhel": ("Red Hat Enterprise Linux", "os", "Red Hat Enterprise Linux"),
    "ubuntu": ("Ubuntu", "os", "Ubuntu"),
    "debian": ("Debian", "os", "Debian"),
    "sles": ("SUSE Linux Enterprise", "os", "SUSE Linux Enterprise"),
    "windows-server": ("Windows Server", "os", "Windows Server"),
    "oracle-linux": ("Oracle Linux", "os", "Oracle Linux"),
    "rocky-linux": ("Rocky Linux", "os", "Rocky Linux"),
    "almalinux": ("AlmaLinux", "os", "AlmaLinux"),
    "centos": ("CentOS", "os", "CentOS"),
    "esxi": ("VMware ESXi", "virtualization", "VMware ESXi"),
    "vcenter": ("VMware vCenter / vSphere", "virtualization", "VMware vSphere"),
    "proxmox-ve": ("Proxmox VE", "virtualization", "Proxmox Virtual Environment"),
    "kubernetes": ("Kubernetes", "containers", "Kubernetes"),
    "red-hat-openshift": ("Red Hat OpenShift", "containers", "OpenShift"),
    "docker-engine": ("Docker Engine", "containers", "Docker (software)"),
    "postgresql": ("PostgreSQL", "database", "PostgreSQL"),
    "mysql": ("MySQL", "database", "MySQL"),
    "mssqlserver": ("Microsoft SQL Server", "database", "Microsoft SQL Server"),
    "oracle-database": ("Oracle Database", "database", "Oracle Database"),
    "mongodb": ("MongoDB", "database", "MongoDB"),
    "redis": ("Redis", "database", "Redis"),
    "mariadb": ("MariaDB", "database", "MariaDB"),
    "netapp-ontap": ("NetApp ONTAP", "storage", "ONTAP"),
    "truenas": ("TrueNAS", "storage", "TrueNAS"),
    "ceph": ("Ceph", "storage", "Ceph (software)"),
    "apache-kafka": ("Apache Kafka", "data-platform", "Apache Kafka"),
    "apache-spark": ("Apache Spark", "data-platform", "Apache Spark"),
    "elasticsearch": ("Elasticsearch", "data-platform", "Elasticsearch"),
}

# Wikipedia sections that are navigation/boilerplate, not knowledge.
DROP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
}

# Chunking
CHUNK_TARGET_WORDS = 180
CHUNK_MAX_WORDS = 260
CHUNK_OVERLAP_SENTENCES = 1
MIN_CHUNK_WORDS = 25

# Retrieval
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5
RRF_K = 60

# Generation (local, via Ollama)
OLLAMA_URL = "http://localhost:11434"
GEN_MODEL = "gemma4:latest"
JUDGE_MODEL = "qwen3:8b"  # different model family from the generator, to avoid self-preference bias
MAX_CONTEXT_WORDS = 1200

# Cost model. Local inference has no API bill; we report (a) an electricity estimate
# from measured wall-clock time and (b) what the same measured token counts would cost on
# a hosted API, so the trade-off is visible. Prices are inputs, edit them here.
LOCAL_POWER_WATTS = 30.0          # approx. M3 SoC package power under sustained LLM load
ELECTRICITY_USD_PER_KWH = 0.17
HOSTED_PRICE_PER_MTOK = {         # USD per 1M tokens (input, output); illustrative inputs
    "hosted-small": (1.00, 5.00),
}
