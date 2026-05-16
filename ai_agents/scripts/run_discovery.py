#!/usr/bin/env python3
"""Agent D ad-hoc discovery driver for ai_agents.

Builds papers.json / candidates.json / discovery_log.json for the AI agents topic
with the strict 2023-2025, >=500 citations, top-venue filter.

NOT a refactor of discover_candidates.py (which is read-only / cloned).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
META_DIR = REPO_ROOT / "ai_agents" / "metadata"

REGISTRY_FILE = META_DIR / "papers.json"
CANDIDATES_FILE = META_DIR / "candidates.json"
LOG_FILE = META_DIR / "discovery_log.json"

USER_AGENT = "wcm-graph-builder/1.0 (mailto: ke.ding@anu.edu.au)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

OPENALEX_BASE = "https://api.openalex.org/works"

MIN_CITATIONS = 500
TARGET_PER_CLAIM = 10
YEAR_RANGE = (2023, 2025)
HOST_RATE_SLEEP = 1.05


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.strip().lower())


def normalise_doi(doi):
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi or None


# Allowlist: full source name + raw_source_name forms.
NATURE_ALLOWED_FULL = {
    "nature", "nature machine intelligence", "nature communications",
    "nature methods", "nature medicine", "nature biotechnology",
    "nature chemistry", "nature materials", "nature physics",
    "nature reviews methods primers",
}
SCIENCE_ALLOWED_FULL = {"science", "science robotics", "science advances"}
CELL_ALLOWED_FULL = {"cell", "cell systems", "cell reports methods"}

VENUE_SUBSTRINGS = [
    "neural information processing systems",
    "neurips",
    "international conference on machine learning",
    "icml ",
    "proceedings of machine learning research",
    "international conference on learning representations",
    "iclr",
    "annual meeting of the association for computational linguistics",
    "association for computational linguistics",
    "empirical methods in natural language processing",
    "emnlp",
    "north american chapter of the association for computational linguistics",
    "naacl",
    "aaai conference on artificial intelligence",
    "aaai ",
    "computer vision and pattern recognition",
    "cvpr",
    "international joint conference on artificial intelligence",
    "ijcai",
    "conference on robot learning",
    "corl",
    "robotics: science and systems",
    "ieee international conference on robotics and automation",
    "icra",
    "journal of machine learning research",
    "transactions on machine learning research",
    "international conference on computer vision",
    "european conference on computer vision",
    "ieee/cvf",
]


def venue_allowed(source_name, raw_source_name=None):
    candidates = []
    if source_name:
        candidates.append(normalise(source_name))
    if raw_source_name:
        candidates.append(normalise(raw_source_name))
    if not candidates:
        return False
    for n in candidates:
        if not n:
            continue
        # arXiv
        if "arxiv" in n:
            return True
        # Nature/Science/Cell families
        if n.startswith("nature"):
            if n in NATURE_ALLOWED_FULL:
                return True
            continue
        if n.startswith("science"):
            if n in SCIENCE_ALLOWED_FULL:
                return True
            continue
        if n.startswith("cell"):
            if n in CELL_ALLOWED_FULL:
                return True
            continue
        for s in VENUE_SUBSTRINGS:
            if s in n:
                return True
    return False


def reconstruct_abstract(inv):
    if not inv:
        return ""
    try:
        positions = []
        for word, idxs in inv.items():
            for i in idxs:
                positions.append((i, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)
    except Exception:
        return ""


class HostRateLimiter:
    def __init__(self, sleep=HOST_RATE_SLEEP):
        self.sleep = sleep
        self.last = {}

    def wait(self, host):
        now = time.monotonic()
        delta = now - self.last.get(host, 0.0)
        if delta < self.sleep:
            time.sleep(self.sleep - delta)
        self.last[host] = time.monotonic()


def request_with_retry(url, limiter, host, attempts=3, timeout=30):
    last = None
    for i in range(attempts):
        limiter.wait(host)
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"http {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(2 ** i)
    sys.stderr.write(f"giving up after {attempts}: {last} {url[:80]}\n")
    return None


def fetch_openalex(query, limiter, per_page=50, strict=True):
    fil = f"is_paratext:false,from_publication_date:{YEAR_RANGE[0]}-01-01,to_publication_date:{YEAR_RANGE[1]}-12-31"
    if strict:
        fil += f",cited_by_count:>{MIN_CITATIONS - 1}"
    params = {
        "search": query,
        "per_page": str(per_page),
        "filter": fil,
        "sort": "cited_by_count:desc",
        "select": (
            "id,doi,title,publication_year,primary_location,locations,"
            "cited_by_count,abstract_inverted_index,authorships,is_paratext,type"
        ),
    }
    url = OPENALEX_BASE + "?" + urllib.parse.urlencode(params)
    data = request_with_retry(url, limiter, "api.openalex.org")
    if not data:
        return []
    return data.get("results", []) or []


def fetch_openalex_id(oid, limiter):
    url = f"{OPENALEX_BASE}/{oid}?select=id,doi,title,publication_year,primary_location,locations,cited_by_count,abstract_inverted_index,authorships,is_paratext,type"
    return request_with_retry(url, limiter, "api.openalex.org")


def get_source_name(work):
    primary = work.get("primary_location") or {}
    src = (primary or {}).get("source") or {}
    raw = (primary or {}).get("raw_source_name") or ""
    nm = src.get("display_name") if src else ""
    return nm, raw


def normalize_openalex(work):
    title = work.get("title") or ""
    if not title:
        return None
    nm, raw = get_source_name(work)
    journal = nm or raw or ""
    if not journal:
        for loc in work.get("locations") or []:
            s = (loc or {}).get("source") or {}
            rs = (loc or {}).get("raw_source_name") or ""
            if s.get("display_name"):
                journal = s["display_name"]
                break
            if rs:
                journal = rs
                break
    doi = normalise_doi(work.get("doi"))
    oid = work.get("id", "")
    if oid.startswith("https://openalex.org/"):
        oid = oid.split("/")[-1]
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    authors = []
    for a in (work.get("authorships") or [])[:25]:
        n = (a.get("author") or {}).get("display_name")
        if n:
            authors.append(n)
    arxiv_id = None
    pl = work.get("primary_location") or {}
    lp = pl.get("landing_page_url") or ""
    m = re.search(r"arxiv\.org/abs/([\w.]+)", lp)
    if m:
        arxiv_id = m.group(1)
    elif doi and doi.startswith("10.48550/arxiv."):
        arxiv_id = doi.replace("10.48550/arxiv.", "")
    raw_src_full = (pl or {}).get("raw_source_name") or ""
    return {
        "title": title,
        "authors": authors,
        "year": work.get("publication_year"),
        "journal": journal,
        "raw_source_name": raw_src_full,
        "doi": doi,
        "openalex_id": oid,
        "arxiv_id": arxiv_id,
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "abstract": abstract,
        "_source": "openalex",
    }


# Per-claim query plans. Each claim has:
#  openalex: list of broad search queries
#  known_openalex_ids: hand-curated landmark IDs to fetch directly
#  positive_terms: agent-relevance markers (paper must contain at least one)
#  negative_terms: drop if title/abstract matches strongly (clinical-only ChatGPT,
#                  pure backbone models, education surveys, etc)
CLAIM_QUERIES = {
    "reasoning-chain-of-thought": {
        "openalex": [
            "tree of thoughts deliberate problem solving",
            "graph of thoughts large language model",
            "self consistency chain of thought reasoning",
            "chain of thought prompting reasoning",
            "step by step reasoning prompt LLM",
            "self-discover composing reasoning structures",
            "tree search language model reasoning",
            "mathematical reasoning large language model",
        ],
        "known_openalex_ids": [
            "W4377130677",  # Tree of Thoughts
        ],
        "positive_terms": [
            "chain of thought", "chain-of-thought", "tree of thoughts",
            "graph of thoughts", "self-consistency", "self consistency",
            "plan-and-solve", "plan and solve", "self-discover", "reasoning",
            "deliberate", "step by step", "step-by-step", "math word problem",
            "prompt", "deliberat",
        ],
        "negative_terms": [
            "esc guideline", "usmle", "medical education", "higher education",
            "swot analysis", "default mode network", "metastasis",
            "clinical practice", "antimicrobial",
        ],
    },
    "reasoning-tool-use": {
        "openalex": [
            "ReAct synergizing reasoning acting language models",
            "Toolformer language models tools",
            "HuggingGPT solving AI tasks ChatGPT Hugging Face",
            "Gorilla large language model connected massive APIs",
            "function calling tool augmented language model",
            "ToolLLM open source LLM mastering APIs",
            "tool learning foundation model",
            "augmented language model survey tools",
            "external tool use language model agent",
            "code interpreter LLM tool",
        ],
        "known_openalex_ids": [
            "W4389991792",  # Coscientist - autonomous chem research tool use
            "W4396723768",  # ChemCrow
        ],
        "positive_terms": [
            "tool", "react", "toolformer", "hugginggpt", "gorilla",
            "function call", "api", "plugin", "augment", "interpreter",
            "execute", "executor", "agent",
        ],
        "negative_terms": [
            "usmle", "esc guideline", "antimicrobial", "metastasis",
            "default mode", "stone age",
        ],
    },
    "reasoning-memory-and-reflection": {
        "openalex": [
            "Reflexion language agents verbal reinforcement learning",
            "Self-Refine iterative refinement self feedback",
            "Self-Discover composing reasoning structures large language model",
            "MemGPT towards LLMs operating systems memory",
            "memory augmented transformer long context agent",
            "self refine self correction language model",
            "long term memory LLM agent",
            "retrieval augmented generation language model",
            "self improvement reasoning iterative",
        ],
        "known_openalex_ids": [
            "W4389984066",  # RAG survey
        ],
        "positive_terms": [
            "reflex", "self-refine", "self refine", "self-discover",
            "memgpt", "memory", "reflection", "self-correct", "self correct",
            "self-improve", "self improve", "self-feedback", "self feedback",
            "iterative refinement", "retrieval augmented", "retrieval-augmented",
            "rag", "long-term", "episodic", "agent",
        ],
        "negative_terms": [
            "usmle", "esc guideline", "antimicrobial", "stone age",
            "metastasis", "negation",
        ],
    },
    "multiagent-role-and-debate": {
        "openalex": [
            "CAMEL communicative agents mind exploration",
            "AutoGen multi-agent conversation framework",
            "MetaGPT meta programming multi-agent collaborative",
            "multi-agent debate factuality reasoning",
            "AgentVerse multi-agent",
            "ChatDev communicative agents software development",
            "role play multi agent large language model",
            "agent collaboration framework LLM",
            "society of mind multi-agent LLM debate",
        ],
        "known_openalex_ids": [],
        "positive_terms": [
            "multi-agent", "multiagent", "multi agent", "debate",
            "role play", "role-play", "camel", "autogen", "metagpt",
            "chatdev", "collaboration", "society", "communicat", "agent",
        ],
        "negative_terms": [
            "antimicrobial", "esc guideline", "default mode", "stone age",
        ],
    },
    "multiagent-autonomous-loops": {
        "openalex": [
            "AutoGPT autonomous task agent",
            "autonomous agent task decomposition",
            "LangGraph state machine agent",
            "agentic workflow long horizon planning",
            "open ended autonomous agent",
            "task planning execution loop language model",
            "agentic AI workflow",
            "autonomous LLM agent loop",
        ],
        "known_openalex_ids": [
            "W4387835442",  # Generative Agents (long-running autonomous loop)
        ],
        "positive_terms": [
            "autonomous", "autogpt", "babyagi", "loop", "workflow",
            "agent", "long-horizon", "long horizon", "planning",
            "execution", "task decomposition", "agentic",
        ],
        "negative_terms": [
            "esc guideline", "usmle", "antimicrobial", "stone age",
        ],
    },
    "multiagent-emergent-behavior": {
        "openalex": [
            "Generative Agents interactive simulacra human behavior",
            "Voyager open-ended embodied agent Minecraft",
            "GITM ghost in Minecraft generally capable agents",
            "JARVIS-1 open world multi-task agents",
            "social simulation language agent population",
            "simulated society LLM agents emergent",
            "Minecraft LLM agent autonomous skill",
            "embodied agent learning open ended",
        ],
        "known_openalex_ids": [
            "W4387835442",  # Generative Agents UIST
        ],
        "positive_terms": [
            "generative agent", "voyager", "minecraft", "embodied",
            "social", "simulation", "emergent", "simulacra",
            "open-ended", "open ended", "open world", "population",
            "agent",
        ],
        "negative_terms": [
            "antimicrobial", "metastasis", "esc guideline",
        ],
    },
    "applications-software-engineering": {
        "openalex": [
            "SWE-bench language models real-world github issues",
            "SWE-agent agent computer interface software engineering",
            "AlphaCode competitive programming code generation",
            "code agent benchmark autonomous github",
            "autonomous program improvement agent",
            "code generation large language model benchmark",
            "repository level code completion",
            "software engineering agent autonomous repair",
            "AgentBench evaluating LLM agents",
        ],
        "known_openalex_ids": [],
        "positive_terms": [
            "swe", "code", "github", "issue", "software", "program",
            "patch", "repository", "agent", "alphacode", "devin",
            "agentbench", "compil", "debug",
        ],
        "negative_terms": [
            "antimicrobial", "esc guideline", "metastasis",
        ],
    },
    "applications-web-and-embodied": {
        "openalex": [
            "WebArena realistic web environment autonomous agents",
            "Mind2Web generalist agent web",
            "VisualWebArena multimodal agents",
            "PaLM-E embodied multimodal language model",
            "RT-2 vision language action models web robotic",
            "RT-1 robotics transformer real-world control",
            "OpenVLA open source vision language action",
            "computer use agent multimodal",
            "WebShop scalable real-world web interaction",
            "GUI agent screen large language model",
        ],
        "known_openalex_ids": [
            "W4385430679",  # RT-1
            "W4366330503",  # LLaVA / Visual Instruction Tuning
        ],
        "positive_terms": [
            "web", "browser", "gui", "screen", "computer use",
            "webarena", "mind2web", "embodied", "robot", "manipulation",
            "vision language action", "vision-language-action", "vla",
            "rt-2", "rt-1", "palm-e", "agent", "visual instruction",
            "multimodal",
        ],
        "negative_terms": [
            "antimicrobial", "esc guideline", "metastasis", "antimicrobial",
        ],
    },
    "applications-scientific-research": {
        "openalex": [
            "ChemCrow augmenting large language models chemistry tools",
            "Coscientist autonomous chemical research large language models",
            "Biomni biomedical agent",
            "GeneAgent gene set biological agent",
            "scientific research agent literature hypothesis",
            "drug discovery agent large language model",
            "automated scientific discovery language model",
            "large language models clinical knowledge encode",
            "foundation model generalist medical artificial intelligence",
            "structured information extraction scientific text",
        ],
        "known_openalex_ids": [
            "W4389991792",  # Coscientist
            "W4396723768",  # ChemCrow
            "W4391836235",  # Structured info extraction
            "W4384071683",  # Med-PaLM (Singhal 2023 Nature)
            "W4406152279",  # Med-PaLM 2 Nature Medicine
            "W4360891289",  # GPT-4 medical challenge problems
            "W4365143687",  # Foundation models for medical AI
            "W4389727268",  # FunSearch
        ],
        "positive_terms": [
            "chemcrow", "coscientist", "biomni", "geneagent", "scientific",
            "research agent", "chemistry", "biology", "biomedical",
            "drug discovery", "hypothesis", "wet-lab", "wet lab",
            "materials", "autonomous lab", "medical", "clinical",
            "agent", "synthesize", "synthesis", "experiment",
            "scientific text", "discovery",
        ],
        "negative_terms": [
            "esc guideline", "usmle exam", "antimicrobial resistance",
            "default mode network", "stone age",
        ],
    },
}

DOMAIN_HINT = {
    "reasoning-chain-of-thought": "reasoning",
    "reasoning-tool-use": "tool-use",
    "reasoning-memory-and-reflection": "memory",
    "multiagent-role-and-debate": "multi-agent",
    "multiagent-autonomous-loops": "autonomous-loop",
    "multiagent-emergent-behavior": "emergent",
    "applications-software-engineering": "code-agent",
    "applications-web-and-embodied": "web-agent",
    "applications-scientific-research": "scientific-agent",
}


def positive_match(rec, terms):
    if not terms:
        return True
    text = ((rec.get("title") or "") + " " + (rec.get("abstract") or "")).lower()
    return any(t in text for t in terms)


def negative_match(rec, terms):
    if not terms:
        return False
    text = ((rec.get("title") or "") + " " + (rec.get("abstract") or "")).lower()
    return any(t in text for t in terms)


def is_review(rec):
    t = (rec.get("title") or "").lower()
    if t.startswith(("review of", "a review of", "review:")):
        return True
    if " survey" in t and ("survey on" in t or "a survey" in t or "survey of" in t):
        return True
    return False


def make_key(rec):
    if rec.get("doi"):
        return f"doi::{rec['doi']}"
    if rec.get("arxiv_id"):
        return f"arx::{rec['arxiv_id']}"
    if rec.get("openalex_id"):
        return f"oa::{rec['openalex_id']}"
    return f"title::{normalise(rec.get('title',''))}"


def main():
    limiter = HostRateLimiter()
    paradigm_map = {
        "reasoning-chain-of-thought": "reasoning",
        "reasoning-tool-use": "reasoning",
        "reasoning-memory-and-reflection": "reasoning",
        "multiagent-role-and-debate": "multiagent",
        "multiagent-autonomous-loops": "multiagent",
        "multiagent-emergent-behavior": "multiagent",
        "applications-software-engineering": "applications",
        "applications-web-and-embodied": "applications",
        "applications-scientific-research": "applications",
    }

    registry = {}
    key_index = {}
    counter = [0]

    def assign_id(rec, discovery_query, domain_hint):
        d = normalise_doi(rec.get("doi"))
        oa = rec.get("openalex_id")
        arx = rec.get("arxiv_id")
        title_key = normalise(rec.get("title") or "")
        keys = []
        if d:
            keys.append(f"doi::{d}")
        if oa:
            keys.append(f"oa::{oa}")
        if arx:
            keys.append(f"arx::{arx}")
        if title_key:
            keys.append(f"title::{title_key}")
        for k in keys:
            if k in key_index:
                return key_index[k]
        counter[0] += 1
        ext = f"AGT-{counter[0]:03d}"
        registry[ext] = {
            "title": rec.get("title"),
            "authors": rec.get("authors") or [],
            "year": rec.get("year"),
            "journal": rec.get("journal"),
            "doi": d,
            "openalex_id": oa,
            "arxiv_id": arx,
            "cited_by_count": int(rec.get("cited_by_count") or 0),
            "abstract": rec.get("abstract") or "",
            "domain": domain_hint,
            "discovery_query": discovery_query,
            "discovered_at": utcnow_iso(),
        }
        for k in keys:
            key_index[k] = ext
        return ext

    candidates_payload = {}
    log_payload = {}

    for claim_id, plan in CLAIM_QUERIES.items():
        sys.stderr.write(f"[discover] {claim_id}\n")
        sys.stderr.flush()
        log = {
            "queries_run": [],
            "raw_total": 0,
            "after_year": 0,
            "after_citations": 0,
            "after_venue": 0,
            "after_relevance": 0,
            "after_dedup": 0,
            "kept_top_n": 0,
            "citation_threshold_attrition": "",
            "notes": "",
        }
        pool = []
        seen = set()
        # OpenAlex broad queries with strict 500-cite filter at-API
        for q in plan.get("openalex", []):
            log["queries_run"].append(f"openalex: {q}")
            works = fetch_openalex(q, limiter, per_page=50, strict=True)
            log["raw_total"] += len(works)
            for w in works:
                rec = normalize_openalex(w)
                if not rec:
                    continue
                k = make_key(rec)
                if k in seen:
                    continue
                seen.add(k)
                pool.append(rec)

        # Direct-fetch known landmark IDs (some are below 500 but we keep
        # them only if they meet the threshold)
        for oid in plan.get("known_openalex_ids", []):
            log["queries_run"].append(f"openalex_id: {oid}")
            w = fetch_openalex_id(oid, limiter)
            if w:
                rec = normalize_openalex(w)
                if rec:
                    k = make_key(rec)
                    if k not in seen:
                        seen.add(k)
                        pool.append(rec)

        # Now apply filters in order: year (already at-API), citation, venue, positive, negative, dedup
        attrition_at_cite = len(pool)
        after_year = [r for r in pool if r.get("year") and YEAR_RANGE[0] <= int(r["year"]) <= YEAR_RANGE[1]]
        log["after_year"] = len(after_year)
        after_cite = [r for r in after_year if int(r.get("cited_by_count") or 0) >= MIN_CITATIONS]
        log["after_citations"] = len(after_cite)
        log["citation_threshold_attrition"] = (
            f"{len(after_year) - len(after_cite)} of {len(after_year)} dropped by >=500 threshold"
        )
        after_venue = [r for r in after_cite if venue_allowed(r.get("journal"), r.get("raw_source_name"))]
        log["after_venue"] = len(after_venue)
        after_pos = [r for r in after_venue if positive_match(r, plan.get("positive_terms", []))]
        after_neg = [r for r in after_pos if not negative_match(r, plan.get("negative_terms", []))]
        log["after_relevance"] = len(after_neg)

        dedup = []
        seen_keys = set()
        review_count = 0
        for r in after_neg:
            k = make_key(r)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            if is_review(r):
                review_count += 1
                if review_count > 1:
                    continue
            dedup.append(r)
        log["after_dedup"] = len(dedup)

        dedup.sort(key=lambda r: (
            -int(r.get("cited_by_count") or 0),
            -int(r.get("year") or 0),
            normalise(r.get("title") or ""),
        ))
        kept = dedup[:TARGET_PER_CLAIM]
        log["kept_top_n"] = len(kept)
        if len(kept) < 5:
            log["notes"] = (
                "Fewer than 5 candidates: OpenAlex cited_by_count for many "
                "agent-classic arXiv preprints (e.g. ReAct, AutoGen, MetaGPT, "
                "Reflexion, SWE-bench, WebArena, Voyager) is below 500 even "
                "though they have thousands of Google-Scholar cites. Per user "
                "instruction the >=500 threshold is NOT lowered."
            )

        candidates = []
        for rank, rec in enumerate(kept, start=1):
            primary_q = plan["openalex"][0] if plan.get("openalex") else ""
            ext = assign_id(rec, discovery_query=primary_q, domain_hint=DOMAIN_HINT[claim_id])
            num = int(ext.split("-")[1])
            cites = int(rec.get("cited_by_count") or 0)
            base = int(round(12 * math.log10(cites + 1))) if cites > 0 else 0
            score = min(100, base + max(0, 25 - 2 * rank))
            candidates.append({
                "id": f"candidate-{claim_id}-AGT{num:03d}",
                "paper_id": ext,
                "claim_ref": claim_id,
                "discovery_score": score,
                "discovery_rank": rank,
                "confidence": "candidate_unscored",
            })
        candidates_payload[claim_id] = {
            "paradigm": paradigm_map[claim_id],
            "candidates": candidates,
        }
        log_payload[claim_id] = log

    registry_sorted = {k: registry[k] for k in sorted(registry.keys())}
    out_registry = {
        "schema_version": "0.1",
        "topic": "ai_agents",
        "generated_at": utcnow_iso(),
        "papers": registry_sorted,
    }
    out_candidates = {
        "schema_version": "0.2",
        "topic": "ai_agents",
        "generated_by": "agent-D",
        "generated_at": utcnow_iso(),
        "claims": candidates_payload,
    }
    out_log = {
        "schema_version": "0.1",
        "topic": "ai_agents",
        "generated_at": utcnow_iso(),
        "citation_threshold": MIN_CITATIONS,
        "year_range": list(YEAR_RANGE),
        "target_per_claim": TARGET_PER_CLAIM,
        "by_claim": log_payload,
        "totals": {
            "unique_papers": len(registry_sorted),
            "claims": len(candidates_payload),
        },
    }

    META_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(out_registry, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with CANDIDATES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(out_candidates, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with LOG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(out_log, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print("=== AI agents discovery ===")
    print(f"Unique AGT papers: {len(registry_sorted)}")
    for cid, c in candidates_payload.items():
        print(f"  {cid}: {len(c['candidates'])} candidates")


if __name__ == "__main__":
    main()
