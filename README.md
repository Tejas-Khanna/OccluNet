# OccluNet
Occlusion-Robust Road Extraction with Network Resilience

> A single-encoder, dual-head segmentation network that turns noisy, occluded satellite imagery into a routable road graph, then stress-tests that graph with graph-theoretic criticality analysis to produce a quantitative **Resilience Index** for flood/accident what-if scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
  - [Network 1 — Encoder / Decoder](#network-1--encoder--decoder)
  - [Network 2 — Graph & Resilience Pipeline](#network-2--graph--resilience-pipeline)
- [How the Model Learns](#how-the-model-learns)
- [How Is This Different From Existing Work?](#how-is-this-different-from-existing-work)
- [Built on Published Foundations](#built-on-published-foundations)
- [Tech Stack](#tech-stack)
- [Architecture Journey — From Complex to Lean](#architecture-journey--from-complex-to-lean)
- [Estimated Implementation Cost](#estimated-implementation-cost)
- [Limitations & Honest Scope](#limitations--honest-scope)
- [Team](#team)
- [License](#license)

---

## Overview

OccluNet-Lite tackles satellite road extraction as a two-network problem instead of a single segmentation pass:

1. **Network 1** takes a 4-channel (RGB + NIR) satellite chip and produces two co-trained outputs — a road-body **segmentation mask** and a 1px-wide **skeleton/centerline** — using an occlusion-aware encoder (ResNet34 + CBAM), a multi-rate dilated bottleneck, and a lightweight recurrent context module (RCPM) at the bottleneck scale.
2. **Network 2** converts those raster predictions into an actual graph: skeletonize → build a graph with `sknw` → heal broken connections with MST/Union-Find → compute betweenness centrality to find "Gatekeeper Nodes" → run sequential node ablation to compute a **Resilience Index** (`R = E_perturbed / E_baseline`).

The result is surfaced through a 5-panel **Gradio dashboard**: input chip → segmentation mask → skeleton overlay → criticality heatmap → live resilience curve.

## Why This Exists

Most road-extraction submissions stop at a segmentation mask. This project goes further — occlusion-aware segmentation → topological healing → graph-theoretic stress testing — so evaluators get a verifiable, quantitative resilience score instead of just a prettier mask.

---

## Architecture

### Network 1 — Encoder / Decoder

```
Input Chip (4ch: RGB + NIR, 512×512)
        │
        ▼
ResNet34 Encoder + CBAM
(4 scales, 64→512 channels, channel + spatial attention)
        │
        ▼
Dilated Bottleneck
(parallel dilation rates 1 / 2 / 4 / 8, concat + project)
        │
        ▼
UNet Decoder + RCPM
(skip connections; strip convolutions + BiGRU @ 64×64
 for long-range road continuity)
        │
   ┌────┴────┐
   ▼         ▼
Head A:    Head B:
Segmentation  Skeleton /
Mask          Centerline
```

**Parameters:** ~24M — trains and runs comfortably on a single Kaggle T4 GPU.

### Network 2 — Graph & Resilience Pipeline

```
Predicted Masks (Segmentation + Skeleton)
        │
        ▼
1. Skeletonize & Merge      — skimage skeletonize OR'd with skeleton-head output
        │
        ▼
2. sknw Graph Construction  — nodes at junctions, edges weighted by length
        │
        ▼
3. MST Gap Healing          — Union-Find bridging of disconnected endpoints
                               under an angle constraint → Connectivity Ratio
        │
        ▼
4. Betweenness Centrality   — weighted centrality identifies top-k
                               "Gatekeeper Nodes" / bottleneck intersections
        │
        ▼
5. Node Ablation             — sequential removal + global-efficiency recompute
                               → Resilience Index R = E(perturbed) / E(baseline)
        │
        ▼
6. Gradio Dashboard           — 5-panel live visualisation for
                               flood / accident what-if scenarios
```

---

## How the Model Learns

Both heads are trained jointly under a composite objective:

| Loss term | Purpose |
|---|---|
| Segmentation (BCE + Dice) | Pixel-accurate road body |
| clDice | Topology-preserving connectivity ([Shit et al., 2021](https://arxiv.org/abs/2003.07311)) |
| Conformity loss | Keeps the skeleton head's prediction inside the segmentation head's predicted mask |
| Edge-aware term | Sharper boundaries under occlusion |

Together these push the model toward high-resolution, topologically connected, perceptually consistent road maps — even under canopy, shadow, or sensor noise.
 uploaded chip, executes the full graph
                                             

---

## How Is This Different From Existing Work?

Benchmarked against established road-extraction architectures working on the same problem class, not just our own earlier attempts:

| Dimension | D-LinkNet (DeepGlobe '18 winner) | CoANet (Mei et al., 2021) | Sat2Graph (graph-tensor prediction) | RoadTracer (iterative agent) | **OccluNet-Lite (Ours)** |
|---|---|---|---|---|---|
| Input type | Single RGB image | Single RGB image | Imagery tiles, large stacks | Imagery + iterative steps | **RGB + NIR, dual-head** |
| Occlusion / noise handling | Dilated center block only, no attention | Attention-gated encoder | Implicit via deep encoder | Low robustness to heavy occlusion | **CBAM (4 scales) + dilated bottleneck + RCPM** |
| Topology preservation | Pixel-wise loss only | Connectivity-aware loss | Predicts graph structure directly | Iterative graph-growing agent | **clDice + skeleton head + conformity loss** |
| Graph extraction | None — raster mask only | None — raster mask only | End-to-end, direct | End-to-end, direct | **Post-hoc sknw + MST healing** |
| Resilience / criticality analysis | None | None | None | None | **Betweenness centrality + node ablation + Resilience Index** |
| Compute footprint | Moderate-heavy, GPU | Moderate-heavy, GPU | Heavy, large training data | Slow iterative inference | **~24M params, T4-friendly** |

No prior architecture in this comparison set combines occlusion-robust segmentation with graph extraction **and** a quantitative resilience/criticality score — that combination is OccluNet-Lite's core contribution.

---

## Built on Published Foundations

Every non-trivial component traces back to a named technique — nothing here is an unverifiable black box.

| OccluNet-Lite Component | Derived From / Inspired By | What We Did Differently |
|---|---|---|
| ResNet34 + CBAM (every encoder scale) | CoANet (Mei et al., 2021) — channel/spatial attention gates noise before it propagates downstream | Applied CBAM at all 4 encoder scales, not a single insertion point |
| Multi-rate dilated bottleneck (rates 1/2/4/8) | D-LinkNet (Zhou et al., 2018, DeepGlobe winner) — dilated center block for long-range road context | Parallel branches at 4 rates, concatenated + projected, vs. D-LinkNet's cascade |
| RCPM — strip convolutions + BiGRU @ 64×64 | Same long-range-continuity goal as Swin Transformer (OARENet lineage) and PathMamba's selective-scan SSM | Cheaper recurrent-over-strips substitute — not true attention or an SSM; lighter-weight, road-specific |
| Dual heads: segmentation + skeleton (joint training) | Sat2Graph and Boeing & Ha — multi-task centerline framing for graph-able outputs | Added explicit conformity loss constraining the skeleton inside the segmentation head's output |
| clDice loss | Shit et al. (2021), *"clDice — A Novel Topology-Preserving Loss"* | Direct, literal implementation — differentiable soft skeletonization via erosion/dilation |
| sknw graph + MST gap healing | RoadTracer (Bastani et al.) and Sat2Graph — same raster-to-routable-graph goal | Classical raster-then-vectorize: skeletonize → sknw → distance/angle-constrained MST bridging, instead of a learned iterative agent or direct graph-tensor prediction |
| Betweenness centrality, node ablation, Resilience Index | General complex-network resilience theory (NetworkX), not the road-extraction literature above | Original addition — no direct analogue in any of the cited segmentation/vectorization papers |

**Honest scope:** we did *not* implement domain-adversarial alignment (as in Topology-Adaptive Domain Adaptation), an iterative graph-growing agent (RoadTracer), or direct graph-tensor prediction (Sat2Graph) — these remain conceptual references, not techniques used in this build.

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, segmentation-models-pytorch (ResNet34-UNet), timm, custom CBAM + clDice modules |
| Data / Geospatial | Rasterio, GDAL, Albumentations, NumPy |
| Graph & Network Analysis | NetworkX, sknw, scikit-image (skeletonize), OSMnx |
| Visualisation / Dashboard | Gradio (primary), Matplotlib; Streamlit + Leaflet/Folium planned upgrade |
| Compute | Kaggle T4 GPU for training; CPU-only for graph post-processing & dashboard |

---

## Architecture Journey — From Complex to Lean

We started broad, diagnosed failure modes with evidence, then converged on a model that actually trains and generalises.

| v1 — Initial Design | Diagnosis — What Broke | v3 — Final: OccluNet-Lite |
|---|---|---|
| 5-channel input: RGB + NIR + pseudo-DSM | All-zero ground-truth masks on several chips | 4-channel input: RGB + NIR |
| Full Swin Transformer encoder stage | 88M-param model trained on only ~300 chips → severe overfit risk | ResNet34 encoder + CBAM at every scale |
| Mamba (state-space) decoder | Incorrect input normalisation | Multi-rate dilated bottleneck (rates 1/2/4/8) |
| Dual decoder heads: road mask + DSM | CUDA OOM during training | UNet decoder + RCPM (strip conv + BiGRU) at 64×64 |
| Occlusion-aware dual-branch fusion block | Swin + Mamba stack too deep for available compute | Dual heads: segmentation + skeleton, trained jointly |
| ~88M parameters | | ~24M parameters, T4-GPU friendly |

Engineering discipline over architectural complexity: we diagnosed exactly why the first design failed and converged on a lean, citable, ablation-tested architecture that gives evaluators a verifiable Resilience Index instead of an unverifiable black box.

---

## Estimated Implementation Cost

| Line Item | POC Cost | Notes |
|---|---|---|
| Software | $0 | Entirely open-source: PyTorch, NetworkX, Gradio, GDAL, etc. |
| Training compute | $0 – $8 | Kaggle free-tier T4 GPU (~70 epochs, <20 GPU-hrs); ~$5–$8 if replicated on paid cloud @ $0.35–$0.50/hr |
| Data & storage | $0 | ~10–15 GB across SpaceNet-5, DeepGlobe, Sentinel-2 — fits Kaggle's free tier |
| Pilot deployment | $0 – $20/mo | Free Hugging Face Spaces tier, or a small VM for a persistent Gradio demo |

**Indicative POC total: under $50.** City-scale processing on Cartosat-3 at production scale would need dedicated GPU capacity and is out of scope for this hackathon estimate.

---

## Limitations & Honest Scope

- No domain-adversarial alignment for cross-sensor transfer (conceptual reference only: Topology-Adaptive Domain Adaptation).
- No iterative graph-growing inference agent (conceptual reference only: RoadTracer).
- No direct graph-tensor prediction (conceptual reference only: Sat2Graph).
- Graph healing (MST/Union-Find) is a classical, post-hoc step rather than a learned one — effective and cheap, but not end-to-end differentiable.
- Resilience Index is a relative, graph-topological measure (`E_perturbed / E_baseline`); it is not calibrated against real-world traffic-flow or disaster-response data.

---

## Team

 Member 
 Member 1 : Tejas Khanna 
 Member 2 : Nehal Agarwal 
 Member 3 : Ayush Panwar
 Member 4 : Vishwas Rajawat

---

