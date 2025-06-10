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
    
    # Flatten nested controller_metrics for easier access
    flattened_data = []
    for entry in data:
        flattened_entry = entry.copy()
        
        # If controller_metrics exists, flatten its contents to top level
        if 'controller_metrics' in entry and isinstance(entry['controller_metrics'], dict):
            controller_metrics = entry['controller_metrics']
            
            # Add each metric to the flattened entry
            for key, value in controller_metrics.items():
                # Avoid overwriting existing top-level keys
                if key not in flattened_entry:
                    flattened_entry[key] = value
        
        flattened_data.append(flattened_entry)
    
    return pd.DataFrame(flattened_data)

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
    """Plot learning rates (handles both standard and geometric training)."""
    plt.figure(figsize=(10, 6))
    
    # Check if this is geometric training (has different field names)
    if 'expert_learning_rate' in df.columns and 'learning_rate' in df.columns:
        # Geometric training format
        plt.plot(df['step'], df['learning_rate'], label='Rotation LR', color='blue')
        plt.plot(df['step'], df['expert_learning_rate'], label='Expert LR', color='red')
    elif 'primary_lr' in df.columns:
        # Standard/Ghost training format
        plt.plot(df['step'], df['primary_lr'], label='Primary LR', color='blue')
        
        # Plot ghost LRs if available
        if 'ghost_lrs' in df.columns:
            ghost_lrs_df = pd.DataFrame(df['ghost_lrs'].tolist(), index=df['step'])
            for i, col in enumerate(ghost_lrs_df.columns):
                plt.plot(ghost_lrs_df.index, ghost_lrs_df[col], label=f'Ghost {i} LR', linestyle=':', alpha=0.8)
    else:
        print("   ⚠️ No learning rate data found for plotting.")
        plt.close()
        return

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
    
    # Plot orthogonality and saturation evolution (with robust column checking)
    has_orthogonality = 'orthogonality_score' in df.columns and not df['orthogonality_score'].isna().all()
    has_saturation = 'saturation_level' in df.columns and not df['saturation_level'].isna().all()
    
    if has_orthogonality and has_saturation:
        # Both metrics available - dual y-axis plot
        ax2.plot(df['step'], df['orthogonality_score'], 
                 label='Orthogonality Score', color='blue', marker='s', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['step'], df['saturation_level'], 
                      label='Saturation Level', color='red', marker='^', linewidth=2)
        
        ax2.set_ylabel('Orthogonality Score', color='blue')
        ax2_twin.set_ylabel('Saturation Level', color='red')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
    elif has_orthogonality:
        # Only orthogonality available
        ax2.plot(df['step'], df['orthogonality_score'], 
                 label='Orthogonality Score', color='blue', marker='s', linewidth=2)
        ax2.set_ylabel('Orthogonality Score')
        ax2.legend()
        
    elif has_saturation:
        # Only saturation available
        ax2.plot(df['step'], df['saturation_level'], 
                 label='Saturation Level', color='red', marker='^', linewidth=2)
        ax2.set_ylabel('Saturation Level')
        ax2.legend()
        
    else:
        # Neither metric available - show placeholder
        ax2.text(0.5, 0.5, 'No orthogonality or saturation data available', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=12, style='italic', color='gray')
        ax2.set_ylabel('No Data')
    
    ax2.set_title('Expert Specialization Dynamics')
    ax2.set_xlabel('Training Step')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rotation_angles_evolution(df, output_path):
    """Plot how rotation angles evolve over training steps (Geometric training)."""
    if 'rotation_angles' not in df.columns:
        print("   ⚠️ No rotation angles data found. Skipping rotation evolution plot.")
        return
    
    # Parse rotation angles data
    rotation_data = []
    for angles_entry in df['rotation_angles']:
        if angles_entry and isinstance(angles_entry, list):
            rotation_data.append(angles_entry)
    
    if not rotation_data:
        print("   ⚠️ Rotation angles data is empty. Skipping rotation evolution plot.")
        return
    
    num_experts = len(rotation_data[0])
    num_dimensions = len(rotation_data[0][0]) if rotation_data[0] else 0
    
    fig, axes = plt.subplots(num_experts, 1, figsize=(12, 6 * num_experts))
    if num_experts == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_dimensions))
    
    for expert_idx in range(num_experts):
        expert_angles = []
        for step_data in rotation_data:
            expert_angles.append(step_data[expert_idx])
        
        expert_angles_df = pd.DataFrame(expert_angles, index=df['step'][:len(expert_angles)])
        
        for dim_idx in range(num_dimensions):
            if dim_idx < expert_angles_df.shape[1]:
                axes[expert_idx].plot(
                    expert_angles_df.index, 
                    expert_angles_df.iloc[:, dim_idx],
                    label=f'θ{dim_idx+1}', 
                    color=colors[dim_idx],
                    marker='o', 
                    linewidth=2,
                    markersize=4
                )
        
        axes[expert_idx].set_title(f'Expert {expert_idx+1} - Rotation Angle Evolution')
        axes[expert_idx].set_xlabel('Training Step')
        axes[expert_idx].set_ylabel('Rotation Angle (radians)')
        axes[expert_idx].legend()
        axes[expert_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_geometric_loss_components(df, output_path):
    """Plot the breakdown of geometric loss components."""
    if 'geometric_components' not in df.columns:
        print("   ⚠️ No geometric loss components data found. Skipping geometric loss plot.")
        return
    
    # Parse geometric components
    components_data = []
    for comp_entry in df['geometric_components']:
        if comp_entry and isinstance(comp_entry, dict):
            components_data.append(comp_entry)
    
    if not components_data:
        print("   ⚠️ Geometric loss components data is empty. Skipping geometric loss plot.")
        return
    
    # Extract component names from first entry
    component_names = list(components_data[0].keys())
    
    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, component in enumerate(component_names):
        values = [comp_data.get(component, 0) for comp_data in components_data]
        steps = df['step'][:len(values)]
        
        plt.plot(steps, values, 
                label=component.replace('_', ' ').title(), 
                color=colors[i % len(colors)],
                marker='o', 
                linewidth=2,
                markersize=4)
    
    plt.title('Geometric Loss Components Evolution')
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_expert_specialization_metrics(df, output_path):
    """Plot expert specialization and rotation efficiency metrics."""
    if 'expert_specialization' not in df.columns and 'rotation_efficiency' not in df.columns:
        print("   ⚠️ No specialization metrics found. Skipping specialization plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot expert specialization
    if 'expert_specialization' in df.columns:
        ax1.plot(df['step'], df['expert_specialization'], 
                label='Expert Specialization', color='blue', 
                marker='s', linewidth=2, markersize=4)
        ax1.set_title('Expert Specialization Evolution')
        ax1.set_ylabel('Specialization Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot rotation efficiency
    if 'rotation_efficiency' in df.columns:
        ax2.plot(df['step'], df['rotation_efficiency'], 
                label='Rotation Efficiency', color='red', 
                marker='^', linewidth=2, markersize=4)
        ax2.set_title('Rotation Efficiency Evolution')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Efficiency Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rotation_pattern_similarity_heatmap(df, output_path):
    """Plot heatmap of rotation pattern similarity between experts (Geometric training)."""
    if 'rotation_angles' not in df.columns:
        print("   ⚠️ No rotation angles data found. Skipping rotation similarity heatmap.")
        return
    
    # Get final rotation angles
    final_rotation_data = None
    for angles_entry in reversed(df['rotation_angles']):
        if angles_entry and isinstance(angles_entry, list):
            final_rotation_data = angles_entry
            break
    
    if not final_rotation_data:
        print("   ⚠️ No valid rotation angles found. Skipping rotation similarity heatmap.")
        return
    
    # Convert to numpy array for easier manipulation
    rotation_matrix = np.array(final_rotation_data)  # [num_experts, num_dimensions]
    num_experts = rotation_matrix.shape[0]
    
    if num_experts < 2:
        print("   ⚠️ Need at least 2 experts for similarity heatmap. Skipping.")
        return
    
    # Compute cosine similarity between expert rotation patterns
    similarity_matrix = np.zeros((num_experts, num_experts))
    
    for i in range(num_experts):
        for j in range(num_experts):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity
                vec_i = rotation_matrix[i]
                vec_j = rotation_matrix[j]
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i, j] = 0.0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f',
                xticklabels=[f'Expert {i+1}' for i in range(num_experts)],
                yticklabels=[f'Expert {i+1}' for i in range(num_experts)],
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title('Rotation Pattern Similarity Matrix\n(Final Training State)')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rotation_magnitude_heatmap(df, output_path):
    """Plot heatmap of rotation magnitudes for each expert and dimension."""
    if 'rotation_angles' not in df.columns:
        print("   ⚠️ No rotation angles data found. Skipping rotation magnitude heatmap.")
        return
    
    # Get final rotation angles
    final_rotation_data = None
    for angles_entry in reversed(df['rotation_angles']):
        if angles_entry and isinstance(angles_entry, list):
            final_rotation_data = angles_entry
            break
    
    if not final_rotation_data:
        print("   ⚠️ No valid rotation angles found. Skipping rotation magnitude heatmap.")
        return
    
    # Convert to numpy array
    rotation_matrix = np.array(final_rotation_data)  # [num_experts, num_dimensions]
    num_experts, num_dimensions = rotation_matrix.shape
    
    plt.figure(figsize=(max(8, num_dimensions * 1.5), max(6, num_experts * 1.2)))
    
    # Create heatmap with rotation magnitudes
    sns.heatmap(rotation_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                xticklabels=[f'θ{i+1}' for i in range(num_dimensions)],
                yticklabels=[f'Expert {i+1}' for i in range(num_experts)],
                cbar_kws={'label': 'Rotation Angle (radians)'})
    
    plt.title('Expert Rotation Angles Heatmap\n(Final Training State)')
    plt.xlabel('Rotation Dimension')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_expert_activation_correlation_heatmap(df, output_path):
    """Plot correlation heatmap of expert activations across training steps."""
    # This requires expert activation data across steps - for now use expert loads as a proxy
    if 'expert_loads' not in df.columns:
        print("   ⚠️ No expert loads data found. Skipping activation correlation heatmap.")
        return
    
    # Extract primary expert activations across all steps
    expert_activations = []
    for loads_entry in df['expert_loads']:
        if loads_entry and isinstance(loads_entry, dict) and 'primary' in loads_entry:
            expert_activations.append(loads_entry['primary'])
    
    if len(expert_activations) < 2:
        print("   ⚠️ Need at least 2 timesteps for correlation analysis. Skipping activation correlation heatmap.")
        return
    
    # Convert to matrix: [timesteps, experts]
    try:
        activation_matrix = np.array(expert_activations)
        num_experts = activation_matrix.shape[1]
        
        if num_experts < 2:
            print("   ⚠️ Need at least 2 experts for correlation analysis. Skipping activation correlation heatmap.")
            return
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activation_matrix.T)  # Transpose to get [experts, timesteps]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f',
                    xticklabels=[f'Expert {i+1}' for i in range(num_experts)],
                    yticklabels=[f'Expert {i+1}' for i in range(num_experts)],
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Expert Activation Correlation Matrix\n(Across Training Steps)')
        plt.xlabel('Expert')
        plt.ylabel('Expert')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️ Error computing activation correlation: {e}")

def detect_training_mode(df, config_data=None):
    """Detect the training mode based on available data columns."""
    # Check config first if available
    if config_data:
        if config_data.get('training_mode') == 'geometric':
            return 'geometric'
        mode = config_data.get('architecture_mode', 'ghost')
        return mode
    
    # Detect based on data columns
    if 'rotation_angles' in df.columns or 'geometric_components' in df.columns:
        return 'geometric'
    elif 'ghost_activations' in df.columns:
        return 'ghost'
    elif 'expert_connections' in df.columns:
        return 'hgnn'
    else:
        return 'standard'

def run_analysis(log_path):
    """
    Main function to load a log and generate all plots based on the run's configuration.
    """
    if not os.path.exists(log_path):
        print(f"❌ ERROR: Log file not found at {log_path}")
        return

    output_dir = os.path.dirname(log_path)
    print(f"📊 Analyzing log file: {log_path}")
    print(f"   Saving plots to: {output_dir}")

    # Load the training log
    df = load_log_data(log_path)
    if df.empty:
        print("   ⚠️ Log file is empty. Skipping analysis.")
        return

    # Load the config to determine which plots are relevant
    config_path = os.path.join(output_dir, "config.json")
    config_data = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        print(f"   ⚠️ config.json not found. Will detect mode from data.")

    # Detect training mode
    mode = detect_training_mode(df, config_data)
    print(f"   Detected training mode: '{mode}'. Generating relevant plots.")

    # Always generate these plots
    plot_losses(df, os.path.join(output_dir, "plot_losses.png"))
    plot_perplexity(df, os.path.join(output_dir, "plot_perplexity.png"))
    plot_learning_rates(df, os.path.join(output_dir, "plot_learning_rates.png"))

    # Plot Geometric-specific plots
    if mode == 'geometric':
        plot_rotation_angles_evolution(df, os.path.join(output_dir, "plot_rotation_angles_evolution.png"))
        plot_geometric_loss_components(df, os.path.join(output_dir, "plot_geometric_loss_components.png"))
        plot_expert_specialization_metrics(df, os.path.join(output_dir, "plot_expert_specialization_metrics.png"))
        
        # Revolutionary geometric-specific heatmaps 🚀
        plot_rotation_pattern_similarity_heatmap(df, os.path.join(output_dir, "plot_rotation_pattern_similarity_heatmap.png"))
        plot_rotation_magnitude_heatmap(df, os.path.join(output_dir, "plot_rotation_magnitude_heatmap.png"))
        plot_expert_activation_correlation_heatmap(df, os.path.join(output_dir, "plot_expert_activation_correlation_heatmap.png"))
        
        # Also plot ghost metrics if available (geometric + ghost combination)
        if 'ghost_activations' in df.columns and 'saturation_level' in df.columns:
            plot_ghost_metrics(df, os.path.join(output_dir, "plot_ghost_metrics.png"))
            plot_expert_load_distribution(df, os.path.join(output_dir, "plot_expert_load_distribution.png"))
            plot_saturation_activation_phase(df, os.path.join(output_dir, "plot_saturation_activation_phase.png"))
            plot_expert_activation_evolution(df, os.path.join(output_dir, "plot_expert_activation_evolution.png"))

    # Plot HGNN-specific plots
    if mode in ['hgnn', 'orthogonal', 'ghost', 'geometric']:
        plot_expert_connection_heatmap(df, os.path.join(output_dir, "plot_expert_connections_heatmap.png"))

    # Plot Ghost-specific plots (for pure ghost mode)
    if mode == 'ghost':
        plot_ghost_metrics(df, os.path.join(output_dir, "plot_ghost_metrics.png"))
        plot_expert_load_distribution(df, os.path.join(output_dir, "plot_expert_load_distribution.png"))
        plot_saturation_activation_phase(df, os.path.join(output_dir, "plot_saturation_activation_phase.png"))
        plot_expert_activation_evolution(df, os.path.join(output_dir, "plot_expert_activation_evolution.png"))
    
    print("   ✅ Analysis and plotting complete.")
