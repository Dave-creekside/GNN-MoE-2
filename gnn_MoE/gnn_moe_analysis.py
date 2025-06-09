#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_analysis.py

Analysis and visualization utilities for GNN-Coupled MoE models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch # Added import

# Assuming GNNMoEConfig and model class GNNMoEModel will be imported in the main script
# from gnn_moe_config import GNNMoEConfig
# from gnn_moe_architecture import GNNMoEModel

def get_save_path(config, base_filename: str, plots_subdir="plots"):
    """Generates a save path for plots, incorporating run_name if available."""
    # Ensure base plots directory exists
    if not os.path.exists(plots_subdir):
        os.makedirs(plots_subdir)
        print(f"📁 Created '{plots_subdir}' directory for output visualizations.")

    prefix = f"{config.run_name}_" if config.run_name else ""
    return os.path.join(plots_subdir, f"{prefix}{base_filename}")

def analyze_expert_communication(model, config, detailed=True):
    print("\n🧠 Expert Communication Analysis")
    # Ensure model.analyze_expert_communication() is a method of GNNMoEModel
    if not hasattr(model, 'analyze_expert_communication'):
        print("⚠️ Model does not have 'analyze_expert_communication' method. Skipping.")
        return None
        
    comm_data = model.analyze_expert_communication()
    if not comm_data: # If the method returns None or empty dict
        print("⚠️ No communication data returned by model.analyze_expert_communication().")
        return None

    for layer_name, matrices_in_layer in comm_data.items():
        print(f"\n{layer_name.upper()}:")
        if not matrices_in_layer: # Check if the list of matrices for this layer is empty
            print(f"  No GNN matrices found for {layer_name}.")
            continue
        for gnn_idx, matrix in enumerate(matrices_in_layer):
            if matrix is None: # Check if a specific matrix is None
                print(f"  GNN Layer {gnn_idx+1} matrix is None.")
                continue
            print(f"  GNN Layer {gnn_idx+1} - Expert connectivity (Adjacency Strength):")
            connectivity = matrix.cpu().numpy()
            if detailed:
                for i in range(connectivity.shape[0]):
                    connections = [f"E{j}:{connectivity[i,j]:.3f}" for j in range(connectivity.shape[1]) if i != j]
                    print(f"    Expert {i} → [{', '.join(connections)}]")
            else:
                print(f"    Avg connectivity: {connectivity.mean():.3f}, Max: {connectivity.max():.3f}")
    return comm_data

def plot_expert_connectivity(comm_data, config):
    save_path = get_save_path(config, "expert_connectivity.png")
    
    if not comm_data:
        print("⚠️ No communication data to plot for expert connectivity.")
        return

    num_model_layers_with_data = len(comm_data)
    if num_model_layers_with_data == 0:
        print("⚠️ Communication data is empty. Cannot plot expert connectivity.")
        return
        
    # Calculate total number of GNN matrices to plot
    total_gnn_matrices = 0
    for layer_name in comm_data:
        if comm_data[layer_name]: # Check if list of matrices is not empty
            total_gnn_matrices += len(comm_data[layer_name])
    
    if total_gnn_matrices == 0:
        print("⚠️ No GNN matrices found in communication data. Cannot plot expert connectivity.")
        return

    # Determine grid size for subplots
    # config.gnn_layers is the number of GNNs in *each* coupler
    cols = min(config.gnn_layers if hasattr(config, 'gnn_layers') else 1, 3) # Max 3 GNN layers per row
    rows = (total_gnn_matrices + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False) # Slightly taller for titles
    axes = axes.flatten()
    plot_idx = 0

    for layer_name, matrices_in_layer in comm_data.items():
        if not matrices_in_layer: continue # Skip if no matrices for this layer
        for gnn_idx, matrix in enumerate(matrices_in_layer):
            if matrix is None: continue # Skip if matrix is None
            if plot_idx >= len(axes): 
                print(f"Warning: Not enough subplot axes for all GNN matrices. Plotted {plot_idx} out of {total_gnn_matrices}.")
                break 
            
            ax = axes[plot_idx]
            connectivity = matrix.cpu().numpy()
            
            im = ax.imshow(connectivity, cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f'{layer_name} GNN-{gnn_idx+1}\n(Avg: {connectivity.mean():.2f})')
            ax.set_xlabel('To Expert'); ax.set_ylabel('From Expert')
            ax.set_xticks(np.arange(config.num_experts))
            ax.set_yticks(np.arange(config.num_experts))
            ax.set_xticklabels([f'E{i}' for i in range(config.num_experts)])
            ax.set_yticklabels([f'E{i}' for i in range(config.num_experts)])
            
            for i in range(config.num_experts):
                for j in range(config.num_experts):
                    ax.text(j, i, f'{connectivity[i,j]:.2f}', ha='center', va='center', 
                            color='white' if connectivity[i,j] > 0.6 else ('black' if connectivity[i,j] > 0.1 else 'grey'), 
                            fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plot_idx += 1
        if plot_idx >= len(axes) and plot_idx < total_gnn_matrices: break # Break outer loop too
    
    for idx_unused in range(plot_idx, len(axes)):
        axes[idx_unused].axis('off')
    
    fig.suptitle(f"Expert Connectivity ({config.run_name if config.run_name else 'DefaultRun'})", fontsize=16, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    plt.savefig(save_path)
    print(f"🎨 Expert connectivity plot saved to {save_path}")
    plt.close(fig)

def plot_training_results(stats, config):
    save_path = get_save_path(config, "training_results.png")
    
    if not stats or not stats['train_loss']: # Basic check if stats are empty
        print("⚠️ No training statistics found to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10)) # Slightly wider
    
    # Training Loss
    axes[0,0].plot(stats['train_loss'], alpha=0.6, label="Train Loss (Raw)")
    if len(stats['train_loss']) > 50:
        # Calculate valid window size for convolution
        window_size = min(50, len(stats['train_loss'])) 
        smoothed_loss = np.convolve(stats['train_loss'], np.ones(window_size)/window_size, mode='valid')
        # Adjust x-axis for smoothed loss
        x_smoothed = np.arange(window_size - 1, len(stats['train_loss']))
        axes[0,0].plot(x_smoothed, smoothed_loss, color='brown', label=f"Smoothed ({window_size} steps)")
    axes[0,0].set_title('Training Loss'); axes[0,0].set_xlabel('Step'); axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

    # Eval Loss and Perplexity
    if stats['eval_step'] and stats['eval_loss'] and stats['eval_perplexity']:
        ax_eval_loss = axes[0,1]
        ax_eval_loss.plot(stats['eval_step'], stats['eval_loss'], 'o-', color='orange', label='Eval Loss', markersize=4)
        ax_eval_loss.set_title('Evaluation Metrics'); ax_eval_loss.set_xlabel('Step'); ax_eval_loss.set_ylabel('Loss', color='orange')
        ax_eval_loss.tick_params(axis='y', labelcolor='orange'); ax_eval_loss.grid(True, alpha=0.3, axis='y') # Grid for primary y-axis

        ax_ppl = ax_eval_loss.twinx()
        ax_ppl.plot(stats['eval_step'], stats['eval_perplexity'], 's-', color='red', label='Eval Perplexity', markersize=4)
        ax_ppl.set_ylabel('Perplexity', color='red'); ax_ppl.tick_params(axis='y', labelcolor='red')
        
        # Combined legend for dual-axis plot
        lines, labels = ax_eval_loss.get_legend_handles_labels()
        lines2, labels2 = ax_ppl.get_legend_handles_labels()
        ax_ppl.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        axes[0,1].set_title('Evaluation Metrics (No Data)')
        axes[0,1].text(0.5, 0.5, "No evaluation data.", ha='center', va='center')


    # Learning Rate
    if stats['learning_rate']:
        axes[1,0].plot(stats['learning_rate'])
        axes[1,0].set_title('Learning Rate Schedule'); axes[1,0].set_xlabel('Step'); axes[1,0].set_ylabel('LR')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].set_title('Learning Rate (No Data)')
        axes[1,0].text(0.5, 0.5, "No LR data.", ha='center', va='center')


    # Gradient Norms
    if stats['grad_norm']:
        axes[1,1].plot(stats['grad_norm'], alpha=0.7, label="Grad Norm")
        axes[1,1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip Threshold (1.0)')
        axes[1,1].set_title('Gradient Norms'); axes[1,1].set_xlabel('Step'); axes[1,1].set_ylabel('Norm')
        axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()
    else:
        axes[1,1].set_title('Gradient Norms (No Data)')
        axes[1,1].text(0.5, 0.5, "No Grad Norm data.", ha='center', va='center')

    
    run_info = config.run_name if config.run_name else 'DefaultRun'
    dataset_info = config.dataset_config_name if hasattr(config, 'dataset_config_name') else 'UnknownDataset'
    fig_title = f"GNN-MoE: {run_info} on {dataset_info}\n({config.num_experts} Experts, {config.gnn_layers} GNN Layers, Emb{config.embed_dim}, {config.num_layers} Layers)"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    fig.suptitle(fig_title, fontsize=14)
    plt.savefig(save_path)
    print(f"📊 Training results plot saved to {save_path}")
    plt.close(fig)

def analyze_model_efficiency(model, config):
    print("\n⚡ GNN-MoE Efficiency Analysis")
    total_params = sum(p.numel() for p in model.parameters())
    
    # Ensure model_layers is the correct attribute name
    model_layers_attr = getattr(model, 'model_layers', None)
    if model_layers_attr is None or not isinstance(model_layers_attr, torch.nn.ModuleList):
        print("Could not find 'model_layers' attribute or it's not a ModuleList. Skipping detailed breakdown.")
        print(f"  Total Parameters: {total_params:,}")
        return

    expert_params = sum(p.numel() for layer in model.model_layers for expert in layer.experts for p in expert.parameters())
    gnn_params = sum(p.numel() for layer in model.model_layers for p in layer.gnn_coupler.parameters())
    other_params = total_params - expert_params - gnn_params
    
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Expert Parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)")
    print(f"  GNN Coord Params: {gnn_params:,} ({gnn_params/total_params*100:.1f}%)")
    print(f"  Other (Embeds, etc.): {other_params:,} ({other_params/total_params*100:.1f}%)")
    print(f"  Expert Utilization: ALL {config.num_experts} experts active per layer.")
    print(f"  GNN Coordination: {config.gnn_layers} GNN layers per expert group.")

if __name__ == '__main__':
    # Example usage (requires GNNMoEConfig and GNNMoEModel to be defined/imported)
    # from gnn_moe_config import GNNMoEConfig
    # from gnn_moe_architecture import GNNMoEModel
    
    print("gnn_moe_analysis.py executed directly. Contains analysis and plotting utilities.")
    
    # Dummy config and stats for testing plotting functions
    @dataclass
    class DummyConfigForAnalysis:
        run_name: str = "test_run"; num_experts: int = 2; gnn_layers: int = 1
        embed_dim: int = 32; num_layers: int = 1; dataset_config_name: str = "dummy_data"

    dummy_cfg = DummyConfigForAnalysis()
    
    dummy_stats = defaultdict(list)
    dummy_stats['train_loss'] = np.random.rand(100) * 5 + 1 # Simulate decreasing loss
    dummy_stats['train_loss'] = np.sort(dummy_stats['train_loss'])[::-1] * np.linspace(1, 0.5, 100)
    dummy_stats['eval_step'] = np.arange(0, 100, 10)
    dummy_stats['eval_loss'] = np.random.rand(10) * 3 + 2
    dummy_stats['eval_perplexity'] = np.exp(dummy_stats['eval_loss'])
    dummy_stats['learning_rate'] = np.linspace(5e-4, 1e-5, 100)
    dummy_stats['grad_norm'] = np.random.rand(100) * 1.5 + 0.5

    print("\nTesting plot_training_results with dummy data...")
    plot_training_results(dummy_stats, dummy_cfg)

    # Dummy communication data for plot_expert_connectivity
    # Requires a GNNMoEModel instance to generate real comm_data
    print("\nTesting plot_expert_connectivity (requires model instance for real data)...")
    # This part would typically be:
    # model = GNNMoEModel(dummy_cfg_for_model) # Assuming a model config
    # comm_data = analyze_expert_communication(model, dummy_cfg_for_model)
    # if comm_data: plot_expert_connectivity(comm_data, dummy_cfg_for_model)
    # For now, just indicate it needs a model.
    print("Skipping plot_expert_connectivity direct test as it needs a model instance.")
