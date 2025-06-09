import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_log_data(log_path):
    """Load data from a single training_log.json file."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_losses(df, output_path):
    """Plot training and evaluation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_loss'], label='Train Loss (Batch)', alpha=0.7)
    plt.plot(df['step'], df['eval_loss'], label='Eval Loss', marker='o', linestyle='--')
    plt.title('Loss vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_perplexity(df, output_path):
    """Plot evaluation perplexity."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['eval_perplexity'], label='Perplexity', marker='o', color='green')
    plt.title('Perplexity vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_learning_rates(df, output_path):
    """Plot primary and ghost learning rates."""
    plt.figure(figsize=(10, 6))
    
    # Plot primary LR
    plt.plot(df['step'], df['primary_lr'], label='Primary LR', color='blue')

    # Plot ghost LRs
    ghost_lrs_df = pd.DataFrame(df['ghost_lrs'].tolist(), index=df['step'])
    for i, col in enumerate(ghost_lrs_df.columns):
        plt.plot(ghost_lrs_df.index, ghost_lrs_df[col], label=f'Ghost {i} LR', linestyle=':', alpha=0.8)

    plt.title('Learning Rates vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_ghost_metrics(df, output_path):
    """Plot ghost activations and saturation level."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot saturation level on the first y-axis
    ax1.plot(df['step'], df['saturation_level'], label='Saturation Level', color='red', linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Saturation Level', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

    # Create a second y-axis for activations
    ax2 = ax1.twinx()
    activations_df = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    for i, col in enumerate(activations_df.columns):
        ax2.plot(activations_df.index, activations_df[col], label=f'Ghost {i} Activation', alpha=0.7)
    
    ax2.set_ylabel('Activation Level')
    ax2.legend(loc='upper right')
    
    plt.title('Ghost Activations and Saturation vs. Steps')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_expert_connection_heatmap(df, output_path):
    """Plot heatmap of expert connections/adjacency matrix."""
    # Get the final expert connections (last logged entry)
    if 'expert_connections' not in df.columns or df['expert_connections'].empty:
        print("   ⚠️ No expert connection data found. Skipping heatmap.")
        return
    
    final_connections = df['expert_connections'].iloc[-1]
    
    if not final_connections:
        print("   ⚠️ Expert connections data is empty. Skipping heatmap.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Try different connection data formats
    if 'adjacency_matrix' in final_connections:
        matrix = np.array(final_connections['adjacency_matrix'])
        title = "Expert Adjacency Matrix"
    elif 'edge_weights' in final_connections:
        weights = final_connections['edge_weights']
        # Convert edge weights to matrix format if needed
        if isinstance(weights, list):
            size = int(np.sqrt(len(weights)))
            matrix = np.array(weights).reshape(size, size)
        else:
            matrix = np.array(weights)
        title = "Expert Edge Weights"
    else:
        print("   ⚠️ Unknown expert connection format. Skipping heatmap.")
        return
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f',
                xticklabels=[f'Expert {i}' for i in range(matrix.shape[1])],
                yticklabels=[f'Expert {i}' for i in range(matrix.shape[0])])
    
    plt.title(f'{title} - Final Training Step')
    plt.xlabel('Target Expert')
    plt.ylabel('Source Expert')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_expert_load_distribution(df, output_path):
    """Plot bar chart of expert loads/activation totals."""
    if 'expert_loads' not in df.columns or df['expert_loads'].empty:
        print("   ⚠️ No expert load data found. Skipping load distribution.")
        return
    
    # Aggregate loads across all time steps
    all_loads = []
    for loads_entry in df['expert_loads']:
        if loads_entry:
            all_loads.append(loads_entry)
    
    if not all_loads:
        print("   ⚠️ Expert load data is empty. Skipping load distribution.")
        return
    
    # Calculate mean loads across time steps
    load_keys = all_loads[0].keys()
    mean_loads = {key: np.mean([load_entry[key] for load_entry in all_loads]) 
                  for key in load_keys}
    
    plt.figure(figsize=(12, 6))
    experts = list(mean_loads.keys())
    loads = list(mean_loads.values())
    
    bars = plt.bar(experts, loads, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, load in zip(bars, loads):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{load:.3f}', ha='center', va='bottom')
    
    plt.title('Expert Load Distribution (Average Across Training)')
    plt.xlabel('Expert')
    plt.ylabel('Average Load')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_saturation_activation_phase(df, output_path):
    """Create scatter plot of saturation vs ghost activation levels."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot for each ghost expert
    activations_df = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    
    colors = plt.cm.Set1(np.linspace(0, 1, activations_df.shape[1]))
    
    for i, col in enumerate(activations_df.columns):
        plt.scatter(df['saturation_level'], activations_df[col], 
                   label=f'Ghost {i}', alpha=0.7, color=colors[i], s=60)
    
    # Add threshold line (assuming 0.01 from your discovery)
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.8, 
                label='Activation Threshold (0.01)')
    
    # Add trajectory arrows showing training progression
    for i, col in enumerate(activations_df.columns):
        if len(df) > 1:
            plt.annotate('', xy=(df['saturation_level'].iloc[-1], activations_df[col].iloc[-1]),
                        xytext=(df['saturation_level'].iloc[0], activations_df[col].iloc[0]),
                        arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.5, lw=2))
    
    plt.xlabel('Saturation Level')
    plt.ylabel('Ghost Activation Level')
    plt.title('Ghost Activation vs Primary Expert Saturation\n(Training Trajectory)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_expert_activation_evolution(df, output_path):
    """Plot how each expert's activation evolves over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot ghost activations
    activations_df = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    for i, col in enumerate(activations_df.columns):
        ax1.plot(activations_df.index, activations_df[col], 
                label=f'Ghost {i}', marker='o', linewidth=2, markersize=4)
    
    ax1.set_title('Ghost Expert Activation Evolution')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Activation Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot orthogonality and saturation evolution
    ax2.plot(df['step'], df['orthogonality_score'], 
             label='Orthogonality Score', color='blue', marker='s', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['step'], df['saturation_level'], 
                  label='Saturation Level', color='red', marker='^', linewidth=2)
    
    ax2.set_title('Expert Specialization Dynamics')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Orthogonality Score', color='blue')
    ax2_twin.set_ylabel('Saturation Level', color='red')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_analysis(log_path):
    """
    Main function to load a log and generate all plots.
    """
    if not os.path.exists(log_path):
        print(f"❌ ERROR: Log file not found at {log_path}")
        return

    output_dir = os.path.dirname(log_path)
    print(f"📊 Analyzing log file: {log_path}")
    print(f"   Saving plots to: {output_dir}")

    df = load_log_data(log_path)
    if df.empty:
        print("   ⚠️ Log file is empty. Skipping analysis.")
        return

    # Original plots
    plot_losses(df, os.path.join(output_dir, "plot_losses.png"))
    plot_perplexity(df, os.path.join(output_dir, "plot_perplexity.png"))
    plot_learning_rates(df, os.path.join(output_dir, "plot_learning_rates.png"))
    plot_ghost_metrics(df, os.path.join(output_dir, "plot_ghost_metrics.png"))
    
    # Enhanced plots
    plot_expert_connection_heatmap(df, os.path.join(output_dir, "plot_expert_connections_heatmap.png"))
    plot_expert_load_distribution(df, os.path.join(output_dir, "plot_expert_load_distribution.png"))
    plot_saturation_activation_phase(df, os.path.join(output_dir, "plot_saturation_activation_phase.png"))
    plot_expert_activation_evolution(df, os.path.join(output_dir, "plot_expert_activation_evolution.png"))
    
    print("   ✅ Analysis and plotting complete with enhanced visualizations.")
