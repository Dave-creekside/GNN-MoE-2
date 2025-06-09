# Build Log: 01 - Phase 1 HGNN Initial Setup (Static "All Pairs")

**Date:** 2025-06-01

## Goal
Integrate an initial Hypergraph Neural Network (HGNN) based expert coupler into the GNN-MoE architecture as an alternative to the existing GNN-based coupler. This serves as the first step towards the nested HGNN-MoE architecture.

## Directory Structure & Setup
- Created experimental directory: `hgnn_moe_dev/`
- Copied core GNN-MoE project files into `hgnn_moe_dev/`.
- Installed PyTorch Geometric and its dependencies (`torch-scatter`, `torch-sparse`).
- Updated `hgnn_moe_dev/requirements.txt` with PyG libraries.
- Created `hgnn_moe_dev/build-log/` for progress tracking.

## Configuration (`gnn_moe_config.py`)
Added new fields to `GNNMoEConfig` in `hgnn_moe_dev/gnn_moe_config.py`:
- `coupler_type: str = "GNN"` (Default, can be set to "HGNN")
- `hgnn_conv_type: Optional[str] = "HypergraphConv"` (Specifies PyG layer)
- `static_hyperedge_strategy: Optional[str] = "all_pairs"` (Initial strategy)
- `hgnn_learnable_edge_weights: bool = True` (To enable learnable weights per hyperedge)

## Architecture (`gnn_moe_architecture.py`)

1.  **`PyGHypergraphConvWrapper(nn.Module)`:**
    *   A wrapper around `torch_geometric.nn.HypergraphConv`.
    *   Takes `in_channels`, `out_channels`, and PyG `HypergraphConv` specific parameters.
    *   Handles basic initialization of the PyG layer.

2.  **`HGNNExpertCoupler(nn.Module)`:**
    *   Replaces `GNNExpertCoupler` when `config.coupler_type == "HGNN"`.
    *   **Hyperedge Generation:**
        *   Implements `generate_static_hyperedges` method.
        *   Currently supports `"all_pairs"` and `"all_triplets"` strategies for `config.static_hyperedge_strategy`.
        *   Generates `_hyperedge_index` (format: `[2, num_connections]`) and `_num_hyperedges`.
    *   **Learnable Hyperedge Weights:**
        *   If `config.hgnn_learnable_edge_weights` is `True` and hyperedges exist, initializes `self.hyperedge_weights = nn.Parameter(torch.randn(self._num_hyperedges))`.
    *   **Forward Pass:**
        *   Reshapes expert inputs `(B, L, E, D)` to `(B*L, E, D)`.
        *   Iterates through the `B*L` dimension, applying the `PyGHypergraphConvWrapper` to each `(E,D)` slice of expert features along with the shared `hyperedge_index` and `hyperedge_weights`.
        *   **Note:** This iteration is a known performance bottleneck and will be a target for future optimization (e.g., using PyG's `Batch` object or more advanced batching techniques if the chosen PyG layer supports them well for this use case).
        *   Uses one or more `PyGHypergraphConvWrapper` layers (controlled by `config.gnn_layers`).
        *   Applies a final `combiner` (Linear + GELU + LayerNorm) similar to `GNNExpertCoupler`.
    *   **Communication Data:** `get_expert_communication_matrices` returns learnable hyperedge weights if they exist.

3.  **`GNNMoELayer(nn.Module)`:**
    *   Modified `__init__` to check `config.coupler_type`.
    *   Instantiates either `HGNNExpertCoupler` or `GNNExpertCoupler` accordingly.
    *   Prints which coupler type is being used.

4.  **`if __name__ == '__main__':` Test Block:**
    *   Updated to include a test case for instantiating and running a forward pass with `coupler_type="HGNN"`, using the "all_pairs" strategy and learnable edge weights. This helps in basic debugging of the new components.

## Next Steps
- Thoroughly test the `HGNNExpertCoupler` with the "all_pairs" strategy using `run_gnn_moe.py`.
- Debug any issues with `hyperedge_index` generation and its usage with `torch_geometric.nn.HypergraphConv`.
- Verify the shape transformations and data flow through the HGNN layers.
- Compare VRAM usage and basic learning capability against the GNN version with a similar number of experts.
- Investigate optimizing the batched application of the PyG `HypergraphConv` layer to avoid the explicit loop over `B*L`.
