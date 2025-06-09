# Build Log: 02 - Phase 1 HGNN Initial Testing Plan

**Date:** 2025-06-01

## I. Goal of This Testing Phase

*   Verify the functional correctness of the new HGNN components (`PyGHypergraphConvWrapper`, `HGNNExpertCoupler`) implemented in `hgnn_moe_dev/gnn_moe_architecture.py`.
*   Ensure the modified `GNNMoEModel` (configurable for HGNN coupler) can be instantiated and can perform forward and backward passes without errors when `coupler_type="HGNN"`.
*   Conduct initial training runs using `hgnn_moe_dev/run_gnn_moe.py` to:
    *   Observe basic learning behavior (loss decrease, perplexity reduction).
    *   Measure and compare VRAM usage of the HGNN coupler against the GNN baseline.
    *   Measure and compare training speed (throughput) against the GNN baseline.
*   Identify and log any bugs, performance bottlenecks (especially the `B*L` loop in `HGNNExpertCoupler`), or unexpected behaviors.
*   Establish a preliminary performance baseline for the static "all_pairs" HGNN coupler.

## II. Test Environment & Setup

1.  **Location:** All code and scripts are within the `/Users/orion/Projects/gnn-moe/notebooks/hgnn_moe_dev/` directory.
2.  **Python Environment:**
    *   A dedicated virtual environment should be used.
    *   All dependencies from `hgnn_moe_dev/requirements.txt` (including PyTorch, PyTorch Geometric, `torch-scatter`, `torch-sparse`) must be installed.
    *   **Log:** Python version, PyTorch version (`torch.__version__`), PyTorch Geometric version (`torch_geometric.__version__`), and CUDA version (if applicable on GPU systems).
3.  **Hardware:**
    *   Initial debugging and unit tests can be performed on macOS (M3).
    *   Comparative performance metrics (VRAM, speed, training runs) should be collected on target GPU hardware (A100 / T4).
4.  **Data:**
    *   **Unit/Integration Tests:** Dummy data or the `if __name__ == '__main__':` block in `gnn_moe_architecture.py` can be used.
    *   **Training Runs:** A consistent, small subset of a standard dataset (e.g., WikiText-2, ~10k-20k training samples, ~1k-2k evaluation samples) to ensure comparability and quick iterations.
    *   **Log:** Specific dataset name, configuration (e.g., `wikitext-2-v1`), and subset sizes used for each training experiment.

## III. Testing Stages & What to Log

### Stage 1: Static Code Review & Unit/Component Tests

*   **Focus:** `hgnn_moe_dev/gnn_moe_architecture.py` and `hgnn_moe_dev/gnn_moe_config.py`.
*   **Actions & Verifications:**
    1.  **Configuration (`gnn_moe_config.py`):**
        *   Confirm new fields (`coupler_type`, `hgnn_conv_type`, `static_hyperedge_strategy`, `hgnn_learnable_edge_weights`) are present with correct defaults.
    2.  **Hyperedge Generation (`HGNNExpertCoupler.generate_static_hyperedges`):**
        *   Manually invoke with `config.static_hyperedge_strategy="all_pairs"` and `num_experts` = 2, 3, 4.
        *   Verify `self._num_hyperedges` (e.g., for N=4, pairs=6).
        *   Verify `self._hyperedge_index` shape (should be `[2, num_connections]`, e.g., for N=4, pairs=6, connections=12, so shape `[2, 12]`) and content (correct node-to-hyperedge mappings).
        *   Test edge cases: `num_experts` = 0, 1 (should result in 0 hyperedges and an empty index).
    3.  **Module Instantiation:**
        *   `PyGHypergraphConvWrapper`: Instantiate with sample `in_channels`, `out_channels`.
        *   `HGNNExpertCoupler`: Instantiate with a test `GNNMoEConfig` (`coupler_type="HGNN"`, `static_hyperedge_strategy="all_pairs"`, `num_experts=4`, `hgnn_learnable_edge_weights=True/False`). Verify `_hyperedge_index` is created and `hyperedge_weights` parameter exists/doesn't exist as expected.
        *   `GNNMoELayer`: Instantiate with the HGNN-configured config. Verify it creates an `HGNNExpertCoupler`.
        *   `GNNMoEModel`: Instantiate with the HGNN-configured config.
    4.  **`HGNNExpertCoupler.forward()` - Single Pass Test:**
        *   Create dummy `expert_outputs` tensor: `torch.randn(B, L, E, D)` (e.g., B=1, L=1, E=4, D=32).
        *   Pass through an instantiated `HGNNExpertCoupler`.
        *   Verify output tensor shape: `(B, L, D)`.
        *   Verify no runtime errors.
        *   If `hgnn_learnable_edge_weights=True`, create a dummy loss, call `loss.backward()`, and check `coupler.hyperedge_weights.grad` is not None.
*   **Logging:** Document results of these checks in this file or a sub-section.

### Stage 2: Integration Test - Full Model Forward/Backward Pass

*   **Focus:** Using the `if __name__ == '__main__':` block in `hgnn_moe_dev/gnn_moe_architecture.py` or a small dedicated test script.
*   **Actions & Verifications:**
    1.  Run the test script with `coupler_type="HGNN"`.
    2.  Ensure the model instantiates, performs a forward pass with dummy `input_ids` and `attention_mask`, computes loss with dummy `labels`, and performs a backward pass without crashing.
    3.  Check for presence of gradients in various parts of the model, especially the HGNN coupler and expert parameters.
*   **Logging:** Success/failure, any tracebacks.

### Stage 3: Initial Training Runs & Comparative Analysis

*   **Focus:** Using `hgnn_moe_dev/run_gnn_moe.py`.
*   **Experiment Configurations:**
    *   **Baseline GNN Run:**
        *   `coupler_type="GNN"`
        *   `num_experts=4` (or other small N for comparison)
        *   `gnn_layers=1` (or 2, matching intended HGNN layers)
        *   Small `embed_dim` (e.g., 256), `num_layers` (e.g., 2-4 model layers).
        *   `batch_size` (e.g., 32).
        *   `epochs=3-5`, `max_batches_per_epoch=100-200`.
        *   `run_name` like "gnn_base_E4_L1coupler"
    *   **HGNN Run (All Pairs):**
        *   `coupler_type="HGNN"`
        *   `static_hyperedge_strategy="all_pairs"`
        *   `hgnn_learnable_edge_weights=True`
        *   Same `num_experts`, `gnn_layers` (for HGNN coupler layers), `embed_dim`, `num_layers`, `batch_size`, `epochs`, `max_batches_per_epoch` as the GNN baseline run.
        *   `run_name` like "hgnn_pairs_E4_L1coupler_learnW"
    *   **(Optional) HGNN Run (All Pairs, No Learnable Weights):**
        *   Same as above, but `hgnn_learnable_edge_weights=False`.
        *   `run_name` like "hgnn_pairs_E4_L1coupler_noW"
    *   **(Optional) HGNN Run (All Triplets):**
        *   If "all_pairs" looks promising, try `static_hyperedge_strategy="all_triplets"`.
        *   `run_name` like "hgnn_triplets_E4_L1coupler_learnW"
*   **Metrics to Collect & Log (for each run):**
    1.  Full `python run_gnn_moe.py ...` command used.
    2.  Successful completion (Yes/No). If No, full traceback and error messages.
    3.  Total training time.
    4.  Peak VRAM Usage (on GPU, using `nvidia-smi` logs or `torch.cuda.max_memory_allocated()`).
    5.  Training speed (tokens/sec or batches/sec from script output).
    6.  Plots for: Training Loss, Eval Loss, Eval Perplexity (saved by `gnn_moe_analysis.py`).
    7.  Final best Eval Perplexity and Eval Loss.
    8.  Values of `hyperedge_weights` (if learnable) at the end of training (from `model.analyze_expert_communication()` output).
*   **Logging:** Create tables comparing these metrics across runs.

## IV. Iteration Based on Findings

*   **Debugging:** Address any crashes, numerical instability (NaNs), or shape mismatches.
*   **Performance Analysis:**
    *   **VRAM:** Is HGNN more memory-efficient than GNN for the same `num_experts`? This is a key goal.
    *   **Speed:** How does the HGNN training speed (especially with the `B*L` loop) compare? If significantly slower, optimization of the `HGNNExpertCoupler.forward()` batching will be a high priority for Phase 1.2.
*   **Learning Capability:**
    *   Does the HGNN model learn effectively (loss decreases, perplexity improves)?
    *   How does its learning compare to the GNN baseline on the small dataset?
*   **Hyperedge Weights:** Do the learnable hyperedge weights show any interesting patterns or convergence?

## V. Key Logging Principles

*   **Version Control:** Commit code changes frequently within the `hgnn_moe_dev` branch/directory.
*   **Reproducibility:** Log exact configurations, dataset versions/subsets, and environment details for each run.
*   **Clarity:** Summarize findings clearly. Note down hypotheses for observed behaviors.
*   **Next Steps:** Conclude each log entry with clear next steps or areas for further investigation.

This detailed plan will be used to guide the testing of the initial HGNN implementation.
