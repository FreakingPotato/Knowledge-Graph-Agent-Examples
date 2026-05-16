#!/usr/bin/env python3
"""Agent D: Diffusion Models topic discovery.

Builds papers.json / candidates.json / discovery_log.json for the
9 claims in ai_agents/metadata/ontology.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
META_DIR = REPO_ROOT / "ai_agents" / "metadata"
ONTOLOGY_FILE = META_DIR / "ontology.json"
REGISTRY_FILE = META_DIR / "papers.json"
CANDIDATES_FILE = META_DIR / "candidates.json"
LOG_FILE = META_DIR / "discovery_log.json"

USER_AGENT = "wcm-graph-builder/1.0 (mailto: ke.ding@anu.edu.au)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

OPENALEX_BASE = "https://api.openalex.org/works"
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_BASE = "http://export.arxiv.org/api/query"

POLITE_SLEEP = 1.05
_last_call: Dict[str, float] = {}


def _polite(host: str) -> None:
    now = time.time()
    last = _last_call.get(host, 0.0)
    delta = now - last
    if delta < POLITE_SLEEP:
        time.sleep(POLITE_SLEEP - delta)
    _last_call[host] = time.time()


def normalize_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).lower().strip()
    return s


# Per-claim queries. Mix curated landmark phrases + thematic recall.
CLAIM_QUERIES = {
    "foundations-score-and-ddpm": [
        "denoising diffusion probabilistic models",
        "Ho denoising diffusion probabilistic",
        "noise conditional score network",
        "score-based generative model gradient data distribution",
        "improved denoising diffusion probabilistic",
        "variational diffusion models",
        "diffusion models beat GANs image synthesis",
        "elucidating design space diffusion-based",
        "generative modeling by estimating gradients",
        "consistency models diffusion",
    ],
    "foundations-continuous-time-sde": [
        "score-based generative modeling stochastic differential equations",
        "denoising diffusion implicit models DDIM",
        "probability flow ODE diffusion",
        "diffusion Schrödinger bridge",
        "continuous time score matching",
        "DPM-Solver fast sampling diffusion",
        "Karras EDM elucidating design space",
        "rectified flow generative",
        "flow matching generative",
        "stochastic interpolants diffusion",
    ],
    "foundations-latent-and-guidance": [
        "high-resolution image synthesis latent diffusion",
        "classifier-free diffusion guidance",
        "stable diffusion latent",
        "GLIDE photorealistic image generation diffusion",
        "Imagen text-to-image diffusion",
        "DALL-E 2 hierarchical diffusion",
        "classifier guidance diffusion Dhariwal",
        "cascaded diffusion models high fidelity",
        "ControlNet diffusion conditional",
        "diffusion autoencoders meaningful decodable",
    ],
    "language-discrete-state-markov": [
        "structured denoising diffusion discrete state",
        "D3PM discrete diffusion text",
        "Diffusion-LM controllable text generation",
        "argmax flows multinomial diffusion",
        "categorical diffusion text",
        "step-unrolled denoising autoencoders SUNDAE",
        "discrete diffusion language model",
        "reparameterized discrete diffusion",
        "self-conditioning discrete diffusion",
        "Plaid diffusion language modeling",
    ],
    "language-score-entropy-masked": [
        "score entropy discrete diffusion SEDD",
        "masked diffusion language model MDLM",
        "simplified masked diffusion language",
        "ratio matching discrete diffusion",
        "absorbing discrete diffusion language",
        "Lou score entropy discrete diffusion",
        "Sahoo masked diffusion language",
        "Shi simplified masked diffusion",
        "concrete score discrete diffusion",
        "any-order autoregressive diffusion",
        "DiffusionBERT masked language model diffusion",
        "discrete diffusion estimating ratios data distribution",
        "soft masked diffusion language model",
        "AR diffusion mixed autoregressive masked",
        "Markov masked language model diffusion text",
    ],
    "language-llm-scale-diffusion": [
        "LLaDA large language diffusion",
        "DiffuLLaMA diffusion language model",
        "scaling diffusion language model billion",
        "BeyondAR non-autoregressive diffusion language",
        "GENIE diffusion text generation",
        "DiffuSeq sequence to sequence diffusion",
        "diffusion of thought reasoning",
        "block diffusion language model",
        "instruct diffusion language fine-tuning",
        "Inception Labs diffusion language",
        "likelihood-based diffusion language",
        "Plaid diffusion language likelihood",
        "diffusion language model many tasks scaling",
        "diffusion language model from autoregressive",
        "scaling diffusion language model adaptation",
    ],
    "biology-protein-structure": [
        "RFdiffusion de novo protein design",
        "Chroma generative model protein",
        "Genie protein backbone diffusion",
        "FrameDiff SE(3) protein backbone diffusion",
        "FoldingDiff protein backbone",
        "ProteinSGM protein structure score",
        "RFdiffusion all-atom",
        "diffusion protein binder design",
        "AlphaFold diffusion structure",
        "RoseTTAFold diffusion design symmetric",
    ],
    "biology-protein-sequence": [
        "EvoDiff protein sequence diffusion",
        "DPLM diffusion protein language model",
        "diffusion language model protein generation",
        "protein sequence discrete diffusion",
        "ProteinGenerator sequence structure diffusion",
        "Alamdari protein generation evolutionary diffusion",
        "ESM diffusion protein",
        "CARP convolutional autoencoding protein",
        "MSA diffusion protein design",
        "Wang DPLM diffusion protein",
    ],
    "biology-dna-rna-nucleotide": [
        "DNADiffusion regulatory DNA design",
        "DiscDiff DNA sequence diffusion",
        "DRAKES discrete diffusion DNA reward",
        "RNAdiffusion RNA design",
        "RNA aptamer diffusion generative",
        "regulatory genomic sequence diffusion design",
        "cis-regulatory element diffusion design",
        "gRNAde RNA design geometric",
        "RFdiffusion nucleic acid",
        "codon optimization diffusion CDS",
    ],
}

# Curated must-have landmark IDs per claim — arxiv IDs (force-fetched if missing).
# These were verified to resolve via OpenAlex DOI lookup with 10.48550/arXiv.XXXX form.
MUST_HAVE_ARXIV = {
    "foundations-score-and-ddpm": [
        "2006.11239",  # DDPM (Ho 2020)
        "1907.05600",  # NCSN (Song 2019)
        "2102.09672",  # Improved DDPM (Nichol/Dhariwal)
        "2105.05233",  # ADM (Dhariwal 2021)
        "2206.00364",  # EDM (Karras 2022)
        "2303.01469",  # Consistency Models (Song 2023)
        "2107.00630",  # Variational Diffusion (Kingma 2021)
    ],
    "foundations-continuous-time-sde": [
        "2011.13456",  # Score SDE (Song 2021)
        "2010.02502",  # DDIM
        "2206.00364",  # EDM
        "2206.00927",  # DPM-Solver
        "2209.03003",  # Rectified Flow (Liu)
        "2210.02747",  # Flow Matching (Lipman)
        "2209.11215",  # Stochastic Interpolants
        "2106.01357",  # Diffusion Schrödinger Bridge
    ],
    "foundations-latent-and-guidance": [
        "2112.10752",  # LDM / Stable Diffusion
        "2207.12598",  # Classifier-Free Guidance (Ho/Salimans)
        "2105.05233",  # Classifier guidance / ADM
        "2112.10741",  # GLIDE
        "2205.11487",  # Imagen
        "2204.06125",  # DALL-E 2 (unCLIP)
        "2106.15282",  # Cascaded Diffusion
        "2302.05543",  # ControlNet
        "2111.15640",  # Diffusion Autoencoders
    ],
    "language-discrete-state-markov": [
        "2107.03006",  # D3PM (Austin 2021)
        "2205.14217",  # Diffusion-LM (Li 2022)
        "2102.05379",  # Argmax flows / multinomial
        "2112.06749",  # SUNDAE
        "2210.16886",  # Self-conditioning / Analog Bits
    ],
    "language-score-entropy-masked": [
        "2310.16834",  # SEDD (Lou 2024)
        "2406.07524",  # MDLM (Sahoo 2024)
        "2406.04329",  # Simplified MDLM (Shi 2024)
        "2310.17567",  # Concrete score
    ],
    "language-llm-scale-diffusion": [
        "2502.09992",  # LLaDA
        "2410.17891",  # DiffuLLaMA / Scaling Diffusion Language Models via Adaptation
        "2308.12219",  # Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning
        "2305.18619",  # Likelihood-Based Diffusion Language Models (Plaid)
        "2305.09515",  # TESS
    ],
    "biology-protein-structure": [
        "2206.04119",  # FrameDiff / SE(3)
        "2305.04120",  # Genie protein backbone
        "2301.12485",  # FoldingDiff
    ],
    "biology-protein-sequence": [
        "2402.18567",  # DPLM
    ],
    "biology-dna-rna-nucleotide": [
        "2402.06079",  # DiscDiff
        "2410.13643",  # DRAKES
        "2406.01794",  # RNAdiffusion / Latent
    ],
}

# Curated must-have DOI landmarks (Nature etc.) — try DOI lookup first.
MUST_HAVE_DOI = {
    "biology-protein-structure": [
        "10.1038/s41586-023-06415-8",  # RFdiffusion (Watson Nature 2023)
        "10.1038/s41586-023-06728-8",  # Chroma (Ingraham Nature 2023)
    ],
    "biology-protein-sequence": [
        "10.1101/2023.09.11.556673",  # EvoDiff (Alamdari bioRxiv 2023)
    ],
    "biology-dna-rna-nucleotide": [
        "10.1101/2024.02.01.578352",  # DNADiffusion
    ],
}


def http_get(url: str, host: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 30) -> Optional[requests.Response]:
    _polite(host)
    try:
        r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
        return None
    except Exception as e:
        print(f"  HTTP error on {url}: {e}", file=sys.stderr)
        return None


def openalex_search(query: str, per_page: int = 50) -> List[Dict]:
    params = {
        "search": query,
        "per-page": per_page,
        "select": "id,doi,title,display_name,authorships,publication_year,primary_location,cited_by_count,abstract_inverted_index,locations,type,best_oa_location",
    }
    r = http_get(OPENALEX_BASE, "openalex", params=params)
    if not r:
        return []
    try:
        return r.json().get("results", [])
    except Exception:
        return []


def openalex_get_by_arxiv(arxiv_id: str) -> Optional[Dict]:
    # OpenAlex indexes arXiv DOIs as 10.48550/arxiv.XXXX.XXXXX
    doi = f"https://doi.org/10.48550/arXiv.{arxiv_id}"
    url = f"{OPENALEX_BASE}/{doi}"
    r = http_get(url, "openalex")
    if r:
        try:
            return r.json()
        except Exception:
            pass
    # fallback: search by title-less ID
    params = {"filter": f"ids.openalex:no", "search": f"arXiv:{arxiv_id}", "per-page": 5}
    r = http_get(OPENALEX_BASE, "openalex", params=params)
    if r:
        try:
            results = r.json().get("results", [])
            for w in results:
                if arxiv_id in str(w.get("doi", "")):
                    return w
            if results:
                return results[0]
        except Exception:
            pass
    return None


def invert_abstract(inv: Optional[Dict]) -> str:
    if not inv:
        return ""
    try:
        positions: List[Tuple[int, str]] = []
        for word, idxs in inv.items():
            for i in idxs:
                positions.append((i, word))
        positions.sort()
        return " ".join(w for _, w in positions)
    except Exception:
        return ""


def extract_arxiv_id(w: Dict) -> str:
    # check doi, locations, primary_location
    doi = (w.get("doi") or "").lower()
    m = re.search(r"arxiv[\./](\d{4}\.\d{4,5}(?:v\d+)?)", doi)
    if m:
        return m.group(1).split("v")[0]
    locs = w.get("locations") or []
    if w.get("primary_location"):
        locs = [w["primary_location"]] + locs
    for loc in locs:
        if not loc:
            continue
        src = loc.get("source") or {}
        if src.get("display_name") == "arXiv (Cornell University)" or "arxiv" in (src.get("display_name") or "").lower():
            url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
            m = re.search(r"(\d{4}\.\d{4,5})", url)
            if m:
                return m.group(1)
    return ""


def venue_name(w: Dict) -> str:
    # primary_location.source.display_name preferred
    if w.get("primary_location") and w["primary_location"].get("source"):
        nm = w["primary_location"]["source"].get("display_name")
        if nm:
            return nm
    if w.get("host_venue") and w["host_venue"].get("display_name"):
        return w["host_venue"]["display_name"]
    locs = w.get("locations") or []
    for loc in locs:
        src = (loc or {}).get("source") or {}
        if src.get("display_name"):
            return src["display_name"]
    return ""


def authors_list(w: Dict) -> List[str]:
    out = []
    for a in w.get("authorships") or []:
        au = (a.get("author") or {}).get("display_name")
        if au:
            out.append(au)
    return out


# Venue normalization for allowlist matching
VENUE_ALIASES = {
    "arxiv": "arXiv",
    "arxiv (cornell university)": "arXiv",
    "biorxiv": "bioRxiv",
    "biorxiv (cold spring harbor laboratory)": "bioRxiv",
    "medrxiv": "medRxiv",
    "medrxiv (cold spring harbor laboratory)": "medRxiv",
    "neural information processing systems": "Advances in Neural Information Processing Systems",
    "neurips": "Advances in Neural Information Processing Systems",
    "advances in neural information processing systems": "Advances in Neural Information Processing Systems",
    "international conference on machine learning": "International Conference on Machine Learning",
    "icml": "International Conference on Machine Learning",
    "international conference on learning representations": "International Conference on Learning Representations",
    "iclr": "International Conference on Learning Representations",
    "computer vision and pattern recognition": "Conference on Computer Vision and Pattern Recognition",
    "ieee/cvf conference on computer vision and pattern recognition": "Conference on Computer Vision and Pattern Recognition",
    "conference on computer vision and pattern recognition": "Conference on Computer Vision and Pattern Recognition",
    "association for computational linguistics": "Annual Meeting of the Association for Computational Linguistics",
    "empirical methods in natural language processing": "Empirical Methods in Natural Language Processing",
    "nature": "Nature",
    "nature methods": "Nature Methods",
    "nature biotechnology": "Nature Biotechnology",
    "nature machine intelligence": "Nature Machine Intelligence",
    "nature communications": "Nature Communications",
    "nature computational science": "Nature Computational Science",
    "nature chemistry": "Nature Chemistry",
    "nature structural & molecular biology": "Nature Structural & Molecular Biology",
    "cell": "Cell",
    "cell systems": "Cell Systems",
    "science": "Science",
    "science advances": "Science Advances",
    "proceedings of the national academy of sciences": "Proceedings of the National Academy of Sciences",
    "pnas": "Proceedings of the National Academy of Sciences",
    "elife": "eLife",
    "molecular systems biology": "Molecular Systems Biology",
    "genome biology": "Genome Biology",
    "genome research": "Genome Research",
    "nucleic acids research": "Nucleic Acids Research",
    "bioinformatics": "Bioinformatics",
    "journal of chemical information and modeling": "Journal of Chemical Information and Modeling",
    "journal of machine learning research": "Journal of Machine Learning Research",
    "transactions on machine learning research": "Transactions on Machine Learning Research",
    "aaai conference on artificial intelligence": "AAAI Conference on Artificial Intelligence",
}


def canonical_venue(raw: str) -> str:
    if not raw:
        return ""
    low = raw.strip().lower()
    if low in VENUE_ALIASES:
        return VENUE_ALIASES[low]
    # heuristic
    for key, val in VENUE_ALIASES.items():
        if key in low:
            return val
    return raw


def venue_allowed(canon: str, allow: List[str]) -> bool:
    if not canon:
        return False
    if canon in allow:
        return True
    return False


def build_paper_entry(w: Dict, query: str) -> Optional[Dict]:
    title = (w.get("display_name") or w.get("title") or "").strip()
    if not title:
        return None
    year = w.get("publication_year") or 0
    doi = (w.get("doi") or "").replace("https://doi.org/", "") if w.get("doi") else ""
    openalex_id = w.get("id") or ""
    arxiv_id = extract_arxiv_id(w)
    abstract = invert_abstract(w.get("abstract_inverted_index"))
    venue_raw = venue_name(w)
    venue_canon = canonical_venue(venue_raw)
    return {
        "title": title,
        "authors": authors_list(w),
        "year": year,
        "journal": venue_canon or venue_raw,
        "doi": doi,
        "openalex_id": openalex_id,
        "arxiv_id": arxiv_id,
        "cited_by_count": w.get("cited_by_count") or 0,
        "abstract": abstract,
        "domain": "",  # filled per-claim later
        "discovery_query": query,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "_venue_raw": venue_raw,
        "_venue_canon": venue_canon,
    }


# Domain heuristics
def classify_domain(claim_id: str, paper: Dict) -> str:
    paradigm = claim_id.split("-")[0]
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    blob = title + " " + abstract
    # Use claim_id as the strongest signal — domain should match the claim's bucket.
    if claim_id == "biology-dna-rna-nucleotide":
        # within this claim, split DNA vs RNA
        if (" rna " in blob or "rna sequence" in blob or "aptamer" in blob or "ribodiffusion" in blob or "rnadiffusion" in blob or "rnagenesis" in blob or "rnaflow" in blob or "evoflow-rna" in blob) and "dna" not in blob[:200]:
            return "rna-diffusion"
        return "dna-diffusion"
    if claim_id == "biology-protein-structure":
        return "protein-structure-diffusion"
    if claim_id == "biology-protein-sequence":
        return "protein-sequence-diffusion"
    # Title-driven heuristics for foundations/language paradigms
    if ("rfdiffusion" in title) or ("protein backbone" in title):
        return "protein-structure-diffusion"
    if paradigm == "biology":
        # fallback for biology paradigm if claim_id didn't match (shouldn't happen)
        if "dna" in blob or "regulatory" in blob:
            return "dna-diffusion"
        if " rna " in blob or "aptamer" in blob:
            return "rna-diffusion"
        if "protein" in blob and ("structure" in blob or "backbone" in blob or "fold" in blob):
            return "protein-structure-diffusion"
        if "protein" in blob:
            return "protein-sequence-diffusion"
        return "interdisciplinary"
    if paradigm == "biology":
        return "interdisciplinary"
    if paradigm == "language":
        return "language-diffusion"
    if "latent diffusion" in blob or "stable diffusion" in blob or "image" in blob:
        return "vision-diffusion"
    if paradigm == "foundations":
        return "foundation-models"
    return "ml-methods"


# Relevance gating — drop pure off-topic (e.g. image-only diffusion with no NLP/bio angle).
def is_relevant_for_claim(claim_id: str, paper: Dict) -> bool:
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    blob = title + " " + abstract
    # Hard exclude obvious off-topic
    OFF_TOPIC_TERMS = (
        "single-cell rna",
        "scrna-seq",
        "covid",
        "alzheimer",
        "cilia",
        "copii",
        "gpcr",
        "synaptic adhesion",
        "bbsome",
        "rectified linear unit",
        "relu",
        "data-driven discovery of partial differential",
        "bayesian non-parametrics",
        "optimization and quantization in gradient symbol",
        "self-attention based progressive generative adversarial",
        "charmm",
        "matthews correlation coefficient",
        "dictionary learning for integrative",
        "multimodal brain tumor",
        "phylogeography",
        "non-parametric estimation of a multivariate",
        "universality classes in nonequilibrium",
        "schiff base in bacteriorhodopsin",
        "tumor-specific antigens",
        "mg/ti multilayers",
        "myristic acid",
        "exciton-exciton annihilation",
        "scenic: single-cell",
        "mamba: linear-time sequence modeling",
        "integrative network alignment",
        "nanoscale imaging of phonon",
        "phonon dynamics",
        "diffusion models in bioinformatics",  # survey, not a methods paper
        "a survey of diffusion models",  # surveys
        "diffusion models: a comprehensive survey",
        "survey of diffusion models in natural language",
        "diffusion model in medical imaging",
        "ldct",
    )
    for off in OFF_TOPIC_TERMS:
        if off in blob:
            return False
    # Must mention diffusion-family terminology
    if "diffus" not in blob and "score-based" not in blob and "score matching" not in blob and "ddpm" not in blob and "denoising" not in blob and "ddim" not in blob:
        if not any(k in blob for k in ("consistency model", "rectified flow", "flow matching", "stochastic interpolant", "dpm-solver", "schrödinger bridge", "schrodinger bridge", "sedd", "mdlm", "llada", "diffullama")):
            return False
    if claim_id.startswith("foundations-"):
        # foundations: must look like methodology paper (not pure application of vision diffusion)
        if claim_id == "foundations-score-and-ddpm":
            return any(k in blob for k in ("ddpm", "denoising diffusion", "score-based", "score matching", "ncsn", "noise conditional", "elucidating", "consistency model", "variational diffusion", "improved denoising", "diffusion model"))
        if claim_id == "foundations-continuous-time-sde":
            return any(k in blob for k in ("sde", "stochastic differential", "ddim", "implicit", "probability flow", "ode", "edm", "elucidating", "rectified flow", "flow matching", "stochastic interpolant", "schrödinger bridge", "schrodinger bridge", "dpm-solver", "fast sampling"))
        if claim_id == "foundations-latent-and-guidance":
            return any(k in blob for k in ("latent diffusion", "stable diffusion", "classifier-free", "classifier free", "guidance", "high-resolution image synthesis", "glide", "imagen", "dalle", "dall-e", "controlnet", "cascaded diffusion", "perceptual compression"))
        return True
    if claim_id.startswith("language-"):
        if not any(k in blob for k in ("text", "language", "token", "discrete", "nlp", "natural language", "categorical", "absorbing", "mask", "llm")):
            return False
        if claim_id == "language-discrete-state-markov":
            return any(k in blob for k in ("d3pm", "discrete", "diffusion-lm", "diffusionlm", "argmax", "multinomial", "categorical", "sundae", "step-unrolled", "reparameter", "self-conditioning", "absorbing", "markov"))
        if claim_id == "language-score-entropy-masked":
            return any(k in blob for k in ("sedd", "score entropy", "mdlm", "masked diffusion", "ratio matching", "simplified masked", "absorbing", "concrete score", "any-order", "diffusionbert", "soft-masked", "soft masked", "estimating the ratios", "diffusion-based language", "diffusion language"))
        if claim_id == "language-llm-scale-diffusion":
            # require both an LLM-scale signal AND a diffusion-language signal
            has_llm = any(k in blob for k in ("llm", "large language", "billion parameter", "1b parameter", "7b parameter", "scaling", "from autoregressive", "from llama", "from gpt", "instruction", "instruct-tuned", "instruction-tuned", "likelihood-based", "pretraining", "pre-training", "many tasks"))
            has_lang_diff = any(k in blob for k in ("llada", "diffullama", "diffuseq", "genie", "beyondar", "non-autoregressive", "diffusion language", "discrete diffusion", "masked diffusion", "block diffusion", "text generation", "text-to-text", "plaid"))
            if not (has_llm and has_lang_diff):
                # allow specific landmark must-have papers
                pid_terms = ("llada", "diffullama", "beyondar", "plaid", "genie diffusion text")
                if not any(p in blob for p in pid_terms):
                    return False
            # exclude bio/protein/molecule/image-only
            if any(k in blob for k in ("protein", "molecule", "image-to-image", "image generation", "speech synthesis", "agentic image", "video generation", "text-to-image")):
                return False
            return True
        return True
    if claim_id.startswith("biology-"):
        if claim_id == "biology-protein-structure":
            return any(k in blob for k in ("protein", "backbone", "rfdiffusion", "chroma", "framediff", "foldingdiff", "structure", "binder", "scaffold", "se(3)", "all-atom"))
        if claim_id == "biology-protein-sequence":
            if not ("protein" in blob or "peptide" in blob or "amino acid" in blob):
                return False
            # require a diffusion+sequence angle
            has_seq = any(k in blob for k in ("sequence", "amino acid", "language model", "discrete diffusion", "inverse folding", "evodiff", "dplm", "carp", "amp-diffusion", "antimicrobial peptide", "peptide"))
            if not has_seq:
                return False
            # exclude pure-structure papers (no sequence/language angle at all)
            if not any(k in blob for k in ("sequence", "amino acid", "language", "discrete", "evodiff", "dplm", "carp", "peptide")):
                return False
            return True
        if claim_id == "biology-dna-rna-nucleotide":
            # must explicitly mention nucleic-acid generation/design
            has_nucleic = any(k in blob for k in ("dna", "rna ", " rna,", "rna-", "rna.", "nucleotide", "regulatory sequence", "regulatory dna", "enhancer", "aptamer", "codon", "cis-regulatory", "nucleic acid", "ribodiffusion", "rnadiffusion", "discdiff", "dnadiffusion", "drakes", "rnagenesis", "promoter", "non-coding"))
            if not has_nucleic:
                return False
            # exclude protein-only papers
            if "rna-binding" in blob:
                return False
            # exclude general protein papers
            if "protein" in blob and not any(k in blob for k in ("dna", "rna", "nucleic", "aptamer", "regulatory", "enhancer", "promoter", "codon", "ribo", "nucleotide")):
                return False
            # exclude "all-atom" protein design that isn't really about DNA/RNA generation
            if "rosettafold all-atom" in blob and "rna" not in blob and "dna" not in blob:
                return False
            return True
        return True
    return True


def main() -> int:
    ontology = json.loads(ONTOLOGY_FILE.read_text())
    allow = ontology["discovery"]["venue_allowlist"]
    candidates_per_claim = ontology["discovery"]["candidates_per_claim"]
    min_citations = ontology["discovery"]["min_citations"]

    paper_registry: Dict[str, Dict] = {}  # DFM-NNN -> entry
    # de-dup keys: normalized title, doi, openalex_id, arxiv_id
    title_to_id: Dict[str, str] = {}
    doi_to_id: Dict[str, str] = {}
    arxiv_to_id: Dict[str, str] = {}
    openalex_to_id: Dict[str, str] = {}
    next_id = 1

    def reserve_id() -> str:
        nonlocal next_id
        pid = f"DFM-{next_id:03d}"
        next_id += 1
        return pid

    def register(paper: Dict, claim_id: str) -> Optional[str]:
        # dedupe — title match is the most reliable across arXiv/journal versions
        norm_t = normalize_title(paper["title"])
        doi = paper.get("doi") or ""
        arx = paper.get("arxiv_id") or ""
        oa = paper.get("openalex_id") or ""
        pid: Optional[str] = None
        if norm_t and norm_t in title_to_id:
            pid = title_to_id[norm_t]
        elif doi and doi.lower() in doi_to_id:
            pid = doi_to_id[doi.lower()]
        elif arx and arx in arxiv_to_id:
            pid = arxiv_to_id[arx]
        elif oa and oa in openalex_to_id:
            pid = openalex_to_id[oa]

        if pid is None:
            pid = reserve_id()
            entry = {k: v for k, v in paper.items() if not k.startswith("_")}
            entry["domain"] = classify_domain(claim_id, paper)
            paper_registry[pid] = entry
        else:
            # If the existing entry is the lower-citation arxiv preprint version
            # and the new one is the published journal version, prefer the journal version.
            existing = paper_registry[pid]
            new_cc = paper.get("cited_by_count") or 0
            old_cc = existing.get("cited_by_count") or 0
            new_venue_strong = (paper.get("_venue_canon") or "") not in ("arXiv", "bioRxiv", "medRxiv")
            old_venue_arxiv = existing.get("journal") in ("arXiv", "bioRxiv", "medRxiv")
            if (new_venue_strong and old_venue_arxiv) or new_cc > old_cc * 2:
                # replace metadata but keep pid
                entry = {k: v for k, v in paper.items() if not k.startswith("_")}
                entry["domain"] = existing.get("domain") or classify_domain(claim_id, paper)
                paper_registry[pid] = entry

        if doi:
            doi_to_id.setdefault(doi.lower(), pid)
        if arx:
            arxiv_to_id.setdefault(arx, pid)
        if oa:
            openalex_to_id.setdefault(oa, pid)
        if norm_t:
            title_to_id.setdefault(norm_t, pid)
        return pid

    log: Dict[str, Dict] = {}
    claim_to_candidates: Dict[str, List[Dict]] = {}

    for paradigm in ontology["paradigms"]:
        for claim in paradigm["claims"]:
            cid = claim["id"]
            print(f"\n=== {cid} ===", file=sys.stderr)
            queries = CLAIM_QUERIES.get(cid, [])
            seen_for_claim: Dict[str, Dict] = {}  # pid -> paper dict
            raw_total = 0
            query_log = []

            for q in queries:
                results = openalex_search(q, per_page=50)
                query_log.append({"source": "openalex", "q": q, "raw": len(results)})
                raw_total += len(results)
                for w in results:
                    paper = build_paper_entry(w, q)
                    if not paper:
                        continue
                    seen_for_claim.setdefault(_paper_key(paper), paper)

            # force-fetch must-have arxiv IDs
            for axid in MUST_HAVE_ARXIV.get(cid, []):
                already = any((p.get("arxiv_id") == axid) for p in seen_for_claim.values())
                if already:
                    continue
                w = openalex_get_by_arxiv(axid)
                if w:
                    paper = build_paper_entry(w, f"must-have arxiv:{axid}")
                    if paper:
                        seen_for_claim.setdefault(_paper_key(paper), paper)
                        query_log.append({"source": "openalex-byarxiv", "q": axid, "raw": 1})

            # force-fetch must-have DOI landmarks
            for doi in MUST_HAVE_DOI.get(cid, []):
                already = any(((p.get("doi") or "").lower() == doi.lower()) for p in seen_for_claim.values())
                if already:
                    continue
                url = f"{OPENALEX_BASE}/https://doi.org/{doi}"
                r = http_get(url, "openalex")
                if r:
                    try:
                        w = r.json()
                    except Exception:
                        w = None
                    if w:
                        paper = build_paper_entry(w, f"must-have doi:{doi}")
                        if paper:
                            seen_for_claim.setdefault(_paper_key(paper), paper)
                            query_log.append({"source": "openalex-bydoi", "q": doi, "raw": 1})

            # Filtering
            after_venue: List[Dict] = []
            for p in seen_for_claim.values():
                v = p.get("_venue_canon") or p.get("journal")
                if venue_allowed(v, allow):
                    after_venue.append(p)

            must_arx = set(MUST_HAVE_ARXIV.get(cid, []))
            must_doi = set(d.lower() for d in MUST_HAVE_DOI.get(cid, []))
            after_cite: List[Dict] = []
            for p in after_venue:
                yr = p.get("year") or 0
                cc = p.get("cited_by_count") or 0
                is_must = (p.get("arxiv_id") in must_arx) or ((p.get("doi") or "").lower() in must_doi)
                if is_must:
                    after_cite.append(p)
                elif yr >= 2025 and cc >= 2:
                    after_cite.append(p)
                elif cc >= min_citations:
                    after_cite.append(p)

            after_relevance: List[Dict] = []
            for p in after_cite:
                if is_relevant_for_claim(cid, p):
                    after_relevance.append(p)

            # Sort by citations desc; ties by year desc
            after_relevance.sort(key=lambda p: (p.get("cited_by_count") or 0, p.get("year") or 0), reverse=True)
            # In-claim title-dedupe BEFORE truncation: keep the highest-cited variant
            seen_titles: set = set()
            deduped: List[Dict] = []
            for p in after_relevance:
                nt = normalize_title(p["title"])
                if nt in seen_titles:
                    continue
                seen_titles.add(nt)
                deduped.append(p)
            # Prioritize must-have landmarks: pull them to the top of the kept list
            must_arx_set = set(MUST_HAVE_ARXIV.get(cid, []))
            must_doi_set = set(d.lower() for d in MUST_HAVE_DOI.get(cid, []))
            def is_must(p: Dict) -> bool:
                if p.get("arxiv_id") in must_arx_set:
                    return True
                if (p.get("doi") or "").lower() in must_doi_set:
                    return True
                return False
            must_papers = [p for p in deduped if is_must(p)]
            other_papers = [p for p in deduped if not is_must(p)]
            slots_left = max(candidates_per_claim - len(must_papers), 0)
            kept = must_papers[:candidates_per_claim] + other_papers[:slots_left]
            # Re-sort kept list by citations (must-haves keep their slot but ordering by citations)
            kept.sort(key=lambda p: (p.get("cited_by_count") or 0, p.get("year") or 0), reverse=True)

            log[cid] = {
                "queries": query_log,
                "raw_total": raw_total,
                "after_venue": len(after_venue),
                "after_cite": len(after_cite),
                "after_relevance": len(after_relevance),
                "kept": len(kept),
            }
            print(f"  raw={raw_total} venue={len(after_venue)} cite={len(after_cite)} rel={len(after_relevance)} kept={len(kept)}", file=sys.stderr)

            cand_rows = []
            seen_pids: set = set()
            for p in kept:
                pid = register(p, cid)
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                cand_rows.append({
                    "candidate_id": f"candidate-{cid}-{pid}",
                    "paper_id": pid,
                    "claim_ref": cid,
                    "paradigm": paradigm["id"],
                    "paper_title": p["title"],
                    "year": p.get("year"),
                    "journal": p.get("journal"),
                    "cited_by_count": p.get("cited_by_count"),
                    "arxiv_id": p.get("arxiv_id"),
                    "doi": p.get("doi"),
                    "openalex_id": p.get("openalex_id"),
                    "domain": paper_registry[pid]["domain"],
                    "discovery_query": p.get("discovery_query"),
                })
            claim_to_candidates[cid] = cand_rows

    # Write outputs
    now_iso = datetime.now(timezone.utc).isoformat()
    papers_doc = {
        "schema_version": "0.1",
        "topic": "ai_agents",
        "generated_at": now_iso,
        "generated_by": "agent-D (topic-discoverer)",
        "papers": paper_registry,
    }
    REGISTRY_FILE.write_text(json.dumps(papers_doc, indent=2, ensure_ascii=False))

    claims_doc: Dict[str, Any] = {}
    for paradigm in ontology["paradigms"]:
        for claim in paradigm["claims"]:
            cid = claim["id"]
            claims_doc[cid] = {
                "paradigm": paradigm["id"],
                "claim_text": claim["claim"],
                "candidates": claim_to_candidates.get(cid, []),
            }
    cand_doc = {
        "schema_version": "0.2",
        "topic": "ai_agents",
        "generated_at": now_iso,
        "generated_by": "agent-D (topic-discoverer)",
        "claims": claims_doc,
    }
    CANDIDATES_FILE.write_text(json.dumps(cand_doc, indent=2, ensure_ascii=False))

    log_doc = {
        "schema_version": "0.1",
        "topic": "ai_agents",
        "generated_at": now_iso,
        "citation_threshold": min_citations,
        "target_per_claim": candidates_per_claim,
        "by_claim": log,
    }
    LOG_FILE.write_text(json.dumps(log_doc, indent=2, ensure_ascii=False))

    # Summary
    print(f"\n=== Summary ===", file=sys.stderr)
    print(f"papers: {len(paper_registry)}", file=sys.stderr)
    for cid, rows in claim_to_candidates.items():
        print(f"  {cid}: {len(rows)} candidates", file=sys.stderr)

    return 0


def _paper_key(paper: Dict) -> str:
    if paper.get("doi"):
        return f"doi:{paper['doi'].lower()}"
    if paper.get("arxiv_id"):
        return f"arx:{paper['arxiv_id']}"
    if paper.get("openalex_id"):
        return f"oa:{paper['openalex_id']}"
    return f"t:{normalize_title(paper['title'])}"


if __name__ == "__main__":
    sys.exit(main())
