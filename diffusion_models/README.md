# Diffusion Models × Natural Language × Genomic Sequences

A self-contained subtree applying the agentic knowledge-graph pipeline to a third topic: the intersection of **diffusion-based generative modelling**, **natural-language tokens**, and **protein / DNA / RNA sequence design**.

## Open the viewer

[**▶ open `graph.html`**](./graph.html) — the interactive knowledge graph (paradigm cards → expandable claims → ranked evidence rows → multi-screenshot modal stacks + 5-dim rubric breakdown + TSV download).

[**📋 open `index.html`**](./index.html) — the project landing page (overview, stats, methodology, preview).

Live on GitHub Pages once enabled: <https://freakingpotato.github.io/Knowledge-Graph-Agent-Examples/diffusion_models/>

## Topic ontology

**3 paradigms × 3 claims = 9 claims**:

| Paradigm | Claim ID | Subtype |
|---|---|---|
| **Continuous diffusion foundations** | `foundations-score-and-ddpm` | Score matching & denoising diffusion |
|  | `foundations-continuous-time-sde` | Continuous-time SDE framework |
|  | `foundations-latent-and-guidance` | Latent-space + classifier-free guidance |
| **Discrete & language diffusion** | `language-discrete-state-markov` | Discrete-state Markov chains for text |
|  | `language-score-entropy-masked` | Score-entropy & masked diffusion |
|  | `language-llm-scale-diffusion` | LLM-scale discrete diffusion |
| **Biological sequence & structure diffusion** | `biology-protein-structure` | Protein backbone & structure diffusion |
|  | `biology-protein-sequence` | Protein-sequence diffusion language models |
|  | `biology-dna-rna-nucleotide` | DNA / RNA nucleotide diffusion |

## Reproducing

```bash
# 1. Discover candidates
python diffusion_models/scripts/discover_candidates.py

# 2. Score per paradigm (3 parallel scorers, one per paradigm)

# 3. Promote candidates to evidence
python diffusion_models/scripts/review_candidates.py

# 4. Rank within each claim
python diffusion_models/scripts/rank_evidence.py --log

# 5. Render paragraph screenshots
python diffusion_models/scripts/generate_screenshots.py --force

# 6. Build the standalone viewer
python diffusion_models/scripts/build_viewer.py
```

Sister corpus repo (linked from the project landing): <https://github.com/FreakingPotato/Diffusion_Model_Paper_Collection>
