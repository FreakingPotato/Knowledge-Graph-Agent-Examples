# Knowledge Graph Agent Examples

A hub of literature knowledge graphs built by the same agentic pipeline.

[**▶ Open the demo hub**](./index.html) — pick any project; click through to the project page; click *Enter the knowledge graph* to open the interactive viewer.

Live on GitHub Pages: <https://freakingpotato.github.io/Knowledge-Graph-Agent-Examples/>

## Projects

| | Project | Topic | Papers | Evidence | Next-steps |
|---|---|---|---:|---:|---:|
| 📊 | [**Multimodal NLP × Genomic Foundation Models**](./multimodal_genomics/index.html) | Combining language models with DNA / RNA foundation models | **125** | 76 | 5 |
| 🧬 | [**Whole-Cell Model Paper Collection**](./whole_cell_model/index.html) | Curated WCM literature; 5 viewer layouts including Hybrid Model Summary. Live explorer hosted in [a sister repo](https://github.com/FreakingPotato/Whole-Cell-Model-Paper-Collection) | **56** | 89 | — |
| 🎨 | [**Diffusion Models × NL × Genomic Sequences**](./diffusion_models/index.html) | Continuous & discrete diffusion across image, text, and protein / DNA / RNA design. Companion paper collection at [Diffusion_Model_Paper_Collection](https://github.com/FreakingPotato/Diffusion_Model_Paper_Collection) | **78** | 67 | 5 |
| 🤖 | [**AI Agents — Classics of 2023–2025**](./ai_agents/index.html) | Most-cited LLM-agent papers from 2023, 2024, 2025 (≥500 citations × top AI venues). Reasoning + multi-agent + applications. | **62** | 61 | 5 |
| 🚧 | *Your next topic* | Drop a new ontology JSON, re-run, new tile lands here | — | — | — |

## How it works

Every project under this repo was built by the same pipeline of single-purpose agents:

```
Architect          → ontology.json
                    ↓
Discoverer         → papers.json + candidates.json   (OpenAlex / arXiv / bioRxiv)
                    ↓
Scorers ×3         → scored_<paradigm>.json          (parallel, one per paradigm)
                    ↓
Reviewer           → evidence.json                   (primary / secondary tiers)
                    ↓
   ┌───────────┬───────────┬───────────┐
   ▼           ▼           ▼           ▼
PDF fetcher  Ranker      Linker (H)   Planning Linker
                          ↓           ↓
                       paragraph PNGs (multi-sentence highlighted)
                          ↓
                       QC checker  → qc_report.{json,md}
                          ↓
                       Auditor    → standalone graph.html
```

Agents communicate through versioned data files only. Each agent's logic is also a CLI script under each project's `scripts/` directory, so any pipeline stage can be re-run deterministically.

## Adding a new project

1. Create a new sub-directory next to `multimodal_genomics/`.
2. Author a topic-specific `metadata/ontology.json` (paradigms, claims, 5-dimension rubric, high-impact venue allowlist).
3. Re-run the standard pipeline scripts (most are topic-agnostic — copy them from an existing project and re-path to the new metadata directory).
4. Build the project landing page (`index.html` modeled on `multimodal_genomics/index.html`).
5. Add a project card to the top-level `index.html` hub.

## Repo layout

```
Knowledge-Graph-Agent-Examples/
├── README.md                       this file
├── index.html                      ← demo hub (start here)
├── .nojekyll                       GitHub Pages serves nested HTML correctly
├── multimodal_genomics/
│   ├── index.html                  project landing page
│   ├── graph.html                  interactive knowledge graph viewer
│   ├── README.md                   per-project reproducibility instructions
│   ├── assets/
│   │   ├── evidence/               1,262 paragraph-screenshot PNGs
│   │   └── qa/                     preview screenshots
│   ├── metadata/                   ontology / papers / candidates / evidence / qc
│   └── scripts/                    pipeline CLI (topic-agnostic)
└── whole_cell_model/
    ├── index.html                  project landing page (overview + stats + preview)
    └── assets/qa/                  preview screenshots
    ─ live explorer + data live in
       https://github.com/FreakingPotato/Whole-Cell-Model-Paper-Collection
       (linked from the project page via the "Enter the knowledge graph →" button)
```

## License

MIT — see individual project READMEs for any third-party data attributions (paper metadata is from OpenAlex / Unpaywall / arXiv / Europe PMC under their respective open licences; PDFs themselves are not redistributed).
