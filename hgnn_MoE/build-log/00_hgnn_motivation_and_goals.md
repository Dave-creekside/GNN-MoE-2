# Build Log: 00 - HGNN Motivation and Architectural Goals

**Date:** 2025-06-01

## 1. Introduction

This document outlines the motivation for transitioning from the current Graph Neural Network (GNN) based expert coupling mechanism to a Hypergraph Neural Network (HGNN) based approach within the Mixture of Experts (MoE) language model. It also summarizes the broader architectural vision involving nested HGNN layers and compression.

## 2. Motivation from Current GNN-MoE Limitations

The existing GNN-routed MoE architecture, while functional, faces challenges:

*   **VRAM Scalability:** The current GNN implementation encounters significant VRAM limitations, notably overflowing a 40GB A100 GPU when attempting to scale to 16 experts. It performs well with 2-8 experts.
*   **Performance Scaling:** Preliminary results have shown an inverse scaling pattern where configurations with fewer experts (e.g., 2 experts achieving ~92.45 perplexity) outperform those with more experts (e.g., 8 experts at ~95.93 perplexity). This suggests the current GNN routing might not be optimally utilizing or specializing a larger number of experts.
*   **Pairwise Interactions:** Standard GNNs primarily model pairwise relationships between nodes (experts). This might be insufficient for capturing more complex, higher-order interactions or coalitions among multiple experts simultaneously.

## 3. Proposed Solution: Hypergraph Neural Networks (HGNNs)

HGNNs offer a potential solution to these limitations by explicitly modeling higher-order relationships through hyperedges, where a single hyperedge can connect an arbitrary number of nodes.

**Key Hypothesized Advantages of HGNNs in this Context:**

*   **Improved Memory Efficiency:** HGNNs can potentially offer better memory complexity. The target is to move from an O(E²) complexity (where E is the number of experts, typical for dense graph representations) towards something like O(H·k) (where H is the number of hyperedges and k is the average hyperedge size), especially if H is much smaller than E². This could be crucial for scaling to 16+ experts within VRAM limits.
*   **Richer Expert Interactions:** HGNNs can model multi-expert coalitions directly, allowing for more sophisticated and potentially more effective routing and specialization strategies.
*   **Dynamic Hyperedge Formation:** (Future Goal) The ability to dynamically form hyperedges during training could allow the model to learn optimal expert groupings based on context or task.

## 4. Target Architectural Vision (Long-Term)

The exploration of HGNNs is the first step towards a more advanced nested architecture:

```
Input → HGNN-MoE₁ → GNN/LoRA Compression → HGNN-MoE₂ → Output
```

*   **Level 1 (HGNN-MoE₁):** Coarse-grained expert coalition routing (e.g., 4 "meta-experts" or groups).
*   **Level 2 (Compression):** A LoRA-style feature space compression/bottleneck layer to manage information flow and reduce dimensionality between stages.
*   **Level 3 (HGNN-MoE₂):** Fine-grained expert specialization within the pathways defined by HGNN-MoE₁ and modulated by the compression layer (e.g., another 4 experts per pathway).
*   **Effective Capacity:** This nested structure aims to achieve an effective capacity of 16+ expert pathways while managing VRAM.

## 5. Expected Outcomes (Long-Term)

*   **Scalability:** Support 16+ effective experts within a 40GB VRAM limit.
*   **Performance:** Improve upon the current best perplexity (around 87-92) by leveraging hierarchical routing and more expressive expert interactions.
*   **Efficiency:** Maintain or improve training times compared to the current 8-expert GNN configuration, despite increased effective expert capacity.
*   **Future Potential:** Enable scaling to 32+ effective expert pathways.

## 6. Immediate Focus (Phase 1)

The immediate next steps involve replacing the current GNN coupler with a foundational HGNN coupler (using static "all-pairs" hyperedges initially) to establish a baseline, verify VRAM benefits, and ensure functional correctness. This is documented in `01_phase1_hgnn_static_pairs_setup.md` and will be tested according to `02_phase1_hgnn_testing_plan.md`.
