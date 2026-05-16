# AI Agents — Classics of 2023–2025

A focused knowledge graph of the most-cited LLM-agent papers from 2023, 2024, and 2025. Strict filters: **year ∈ {2023, 2024, 2025}**, **cited_by_count ≥ 500**, top AI venues only (NeurIPS / ICML / ICLR / ACL / EMNLP / NAACL / AAAI / CVPR / IJCAI / CoRL / TMLR / JMLR + arXiv).

## Open the viewer

[**▶ `graph.html`**](./graph.html) — interactive knowledge graph (paradigm cards → expandable claims → ranked evidence rows → multi-screenshot modal stacks + 5-dim rubric breakdown + TSV download).

Includes a **🧭 Planning** toggle that surfaces 5 forward-looking research-direction cards with evidence pulled from the Discussion / Limitations / Future-Work sections of the corpus.

[**📋 `index.html`**](./index.html) — project landing page.

Live: <https://freakingpotato.github.io/Knowledge-Graph-Agent-Examples/ai_agents/>

## Ontology

**3 paradigms × 3 claims = 9 claims**:

| Paradigm | Claim ID | Subtype |
|---|---|---|
| **Reasoning, planning, tool use, memory** | `reasoning-chain-of-thought` | Chain / Tree / Graph of Thought |
| | `reasoning-tool-use` | Tool use & function calling |
| | `reasoning-memory-and-reflection` | Memory & reflection |
| **Multi-agent collaboration & emergence** | `multiagent-role-and-debate` | Role-playing & debate |
| | `multiagent-autonomous-loops` | Autonomous agent loops |
| | `multiagent-emergent-behavior` | Emergent behaviour in simulated societies |
| **Real-world agentic applications** | `applications-software-engineering` | Software-engineering agents |
| | `applications-web-and-embodied` | Web, GUI, embodied |
| | `applications-scientific-research` | Scientific research agents |

Plus **5 Planning next-step claims**:
- Persistent autonomous agents
- Self-improving agents
- Verifiable and safe agents
- Multimodal embodied agents
- Scientific discovery agents

## Reproducing

```bash
python ai_agents/scripts/discover_candidates.py    # uses strict 500-cite / 2023+ / top-venue filter
python ai_agents/scripts/review_candidates.py
python ai_agents/scripts/rank_evidence.py --log
python ai_agents/scripts/generate_screenshots.py --force
python ai_agents/scripts/render_planning.py        # for the Planning view
python ai_agents/scripts/build_viewer.py
```
