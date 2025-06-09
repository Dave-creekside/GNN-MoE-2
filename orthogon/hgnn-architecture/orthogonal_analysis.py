#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orthogonal_analysis.py

Analysis utilities for monitoring and visualizing orthogonal expert training.
Provides tools to track expert specialization, compute orthogonality metrics,
and generate visualization reports.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime

def compute_expert_similarity_matrix(expert_outputs: torch.Tensor, method: str = "cosine") -> torch.Tensor:
    """
    Compute similarity matrix between expert outputs.
    
    Args:
        expert_outputs: (B, L, E, D) or (E, D) tensor of expert representations
        method: "cosine", "dot", "euclidean"
    Returns:
        (E, E) similarity matrix
    """
    if expert_outputs.dim() == 4:  # (B, L, E, D)
        # Average across batch and sequence dimensions
        expert_means = expert_outputs.mean(dim=(0, 1))  # (E, D)
    elif expert_outputs.dim() == 2:  # (E, D)
        expert_means = expert_outputs
    else:
        raise ValueError(f"Expected 2D or 4D tensor, got {expert_outputs.dim()}D")
    
    E = expert_means.shape[0]
    
    if method == "cosine":
        # Normalize and compute cosine similarity
        expert_norms = F.normalize(expert_means, p=2, dim=1)
        similarity_matrix = torch.mm(expert_norms, expert_norms.T)
    elif method == "dot":
        # Dot product similarity
        similarity_matrix = torch.mm(expert_means, expert_means.T)
    elif method == "euclidean":
        # Negative euclidean distance (higher = more similar)
        expert_expanded = expert_means.unsqueeze(1)  # (E, 1, D)
        expert_expanded_t = expert_means.unsqueeze(0)  # (1, E, D)
        distances = torch.norm(expert_expanded - expert_expanded_t, dim=2)  # (E, E)
        similarity_matrix = -distances  # Negate so higher = more similar
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity_matrix

def compute_orthogonality_metrics(expert_outputs: torch.Tensor) -> Dict[str, float]:
    """
    Compute various orthogonality metrics for expert outputs.
    
    Args:
        expert_outputs: (B, L, E, D) or (E, D) tensor of expert representations
    Returns:
        Dictionary of orthogonality metrics
    """
    if expert_outputs.dim() == 4:  # (B, L, E, D)
        expert_means = expert_outputs.mean(dim=(0, 1))  # (E, D)
    else:
        expert_means = expert_outputs
    
    E = expert_means.shape[0]
    
    # Compute Gram matrix
    gram_matrix = torch.mm(expert_means, expert_means.T)
    identity_target = torch.eye(E, device=expert_means.device)
    
    # Metrics
    metrics = {}
    
    # 1. Gram matrix deviation from identity
    metrics['gram_identity_mse'] = F.mse_loss(gram_matrix, identity_target).item()
    
    # 2. Off-diagonal magnitude (should be small for orthogonal experts)
    off_diagonal_mask = ~torch.eye(E, dtype=torch.bool, device=expert_means.device)
    off_diagonal_values = gram_matrix[off_diagonal_mask]
    metrics['off_diagonal_mean'] = torch.mean(torch.abs(off_diagonal_values)).item()
    metrics['off_diagonal_std'] = torch.std(off_diagonal_values).item()
    metrics['off_diagonal_max'] = torch.max(torch.abs(off_diagonal_values)).item()
    
    # 3. Diagonal values (should be close to 1 for normalized experts)
    diagonal_values = torch.diagonal(gram_matrix)
    metrics['diagonal_mean'] = torch.mean(diagonal_values).item()
    metrics['diagonal_std'] = torch.std(diagonal_values).item()
    
    # 4. Cosine similarity metrics
    cosine_sim = compute_expert_similarity_matrix(expert_means, method="cosine")
    cosine_off_diagonal = cosine_sim[off_diagonal_mask]
    metrics['cosine_off_diagonal_mean'] = torch.mean(torch.abs(cosine_off_diagonal)).item()
    metrics['cosine_off_diagonal_max'] = torch.max(torch.abs(cosine_off_diagonal)).item()
    
    # 5. Expert norm diversity (experts should have similar magnitudes)
    expert_norms = torch.norm(expert_means, dim=1)
    metrics['norm_mean'] = torch.mean(expert_norms).item()
    metrics['norm_std'] = torch.std(expert_norms).item()
    metrics['norm_coefficient_variation'] = (torch.std(expert_norms) / torch.mean(expert_norms)).item()
    
    # 6. Effective rank (higher = more diverse expert representations)
    # Using nuclear norm approximation
    U, S, V = torch.svd(expert_means)
    metrics['effective_rank'] = (torch.sum(S) ** 2 / torch.sum(S ** 2)).item()
    metrics['singular_value_entropy'] = compute_entropy(S / torch.sum(S))
    
    return metrics

def compute_entropy(probabilities: torch.Tensor) -> float:
    """Compute entropy of a probability distribution."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = probabilities + epsilon
    return -torch.sum(p * torch.log(p)).item()

def analyze_expert_specialization_trajectory(stats: Dict, window_size: int = 50) -> Dict:
    """
    Analyze how expert specialization evolves during training.
    
    Args:
        stats: Training statistics dictionary
        window_size: Window size for smoothing metrics
    Returns:
        Analysis results
    """
    analysis = {
        'orthogonality_trend': {},
        'loss_correlation': {},
        'convergence_metrics': {}
    }
    
    if 'expert_specialization' not in stats or not stats['expert_specialization']:
        return analysis
    
    # Extract specialization metrics over time
    steps = []
    ortho_losses = []
    
    for entry in stats['expert_specialization']:
        step = entry['step']
        metrics = entry['metrics']
        
        steps.append(step)
        ortho_losses.append(metrics.get('total_orthogonality_loss', 0.0))
    
    if len(steps) < 2:
        return analysis
    
    steps = np.array(steps)
    ortho_losses = np.array(ortho_losses)
    
    # Smooth the metrics
    def smooth(values, window):
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window)/window, mode='valid')
    
    if len(ortho_losses) >= window_size:
        smooth_ortho = smooth(ortho_losses, window_size)
        smooth_steps = steps[:len(smooth_ortho)]
        
        # Trend analysis
        if len(smooth_ortho) > 1:
            trend_slope = np.polyfit(smooth_steps, smooth_ortho, 1)[0]
            analysis['orthogonality_trend']['slope'] = float(trend_slope)
            analysis['orthogonality_trend']['final_loss'] = float(smooth_ortho[-1])
            analysis['orthogonality_trend']['reduction_percentage'] = float(
                (smooth_ortho[0] - smooth_ortho[-1]) / smooth_ortho[0] * 100
            ) if smooth_ortho[0] != 0 else 0.0
    
    # Convergence analysis
    if len(ortho_losses) >= 20:
        last_20_steps = ortho_losses[-20:]
        analysis['convergence_metrics']['recent_std'] = float(np.std(last_20_steps))
        analysis['convergence_metrics']['recent_mean'] = float(np.mean(last_20_steps))
        analysis['convergence_metrics']['is_converging'] = float(np.std(last_20_steps)) < 0.01
    
    return analysis

def plot_expert_similarity_heatmap(similarity_matrix: torch.Tensor, 
                                  title: str = "Expert Similarity Matrix",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot heatmap of expert similarity matrix.
    
    Args:
        similarity_matrix: (E, E) similarity matrix
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    Returns:
        matplotlib Figure object
    """
    plt.figure(figsize=figsize)
    
    # Convert to numpy for plotting
    sim_np = similarity_matrix.detach().cpu().numpy()
    
    # Create heatmap
    sns.heatmap(sim_np, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                xticklabels=[f'Expert {i}' for i in range(len(sim_np))],
                yticklabels=[f'Expert {i}' for i in range(len(sim_np))])
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_orthogonality_training_curves(stats: Dict, 
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot training curves showing orthogonality loss evolution.
    
    Args:
        stats: Training statistics dictionary
        save_path: Path to save the plot
        figsize: Figure size
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract training data
    steps = list(range(len(stats.get('train_loss', []))))
    train_loss = stats.get('train_loss', [])
    lm_loss = stats.get('lm_loss', [])
    ortho_loss = stats.get('orthogonality_loss', [])
    
    # Plot 1: Combined losses
    if train_loss and lm_loss and ortho_loss:
        axes[0, 0].plot(steps, train_loss, label='Total Loss', alpha=0.7)
        axes[0, 0].plot(steps, lm_loss, label='LM Loss', alpha=0.7)
        axes[0, 0].plot(steps, ortho_loss, label='Orthogonality Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Orthogonality loss only (smoothed)
    if ortho_loss:
        # Smooth the orthogonality loss
        window = min(50, len(ortho_loss) // 10)
        if window > 1:
            smooth_ortho = np.convolve(ortho_loss, np.ones(window)/window, mode='valid')
            smooth_steps = steps[:len(smooth_ortho)]
            axes[0, 1].plot(smooth_steps, smooth_ortho, 'r-', linewidth=2)
        else:
            axes[0, 1].plot(steps, ortho_loss, 'r-', linewidth=2)
        
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Orthogonality Loss')
        axes[0, 1].set_title('Orthogonality Loss (Smoothed)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Evaluation metrics
    eval_steps = stats.get('eval_step', [])
    eval_loss = stats.get('eval_loss', [])
    eval_ppl = stats.get('eval_perplexity', [])
    
    if eval_steps and eval_loss:
        axes[1, 0].plot(eval_steps, eval_loss, 'g-o', markersize=3)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Evaluation Loss')
        axes[1, 0].set_title('Evaluation Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Perplexity
    if eval_steps and eval_ppl:
        axes[1, 1].plot(eval_steps, eval_ppl, 'b-o', markersize=3)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Evaluation Perplexity')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_orthogonality_report(model, stats: Dict, output_dir: str, 
                                config, sample_input: Optional[torch.Tensor] = None) -> str:
    """
    Generate comprehensive orthogonality analysis report.
    
    Args:
        model: Trained model
        stats: Training statistics
        output_dir: Directory to save report and plots
        config: Model configuration
        sample_input: Optional sample input for forward pass analysis
    Returns:
        Path to generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"orthogonality_report_{timestamp}.html")
    
    # Generate plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training curves
    training_curves_path = os.path.join(plots_dir, "training_curves.png")
    plot_orthogonality_training_curves(stats, save_path=training_curves_path)
    plt.close()
    
    # Analyze current expert specialization
    expert_analysis = {}
    if sample_input is not None:
        model.eval()
        with torch.no_grad():
            # Forward pass to get expert outputs
            model(sample_input)
            
            # Collect expert outputs from each layer
            for i, layer in enumerate(model.model_layers):
                if hasattr(layer, '_last_orthogonality_loss'):
                    # Get expert similarity for this layer
                    # Note: This is a simplified analysis. For full analysis,
                    # we'd need to modify the model to capture expert outputs
                    expert_analysis[f'layer_{i}'] = {
                        'orthogonality_loss': layer.get_last_orthogonality_loss().item()
                    }
    
    # Trajectory analysis
    trajectory_analysis = analyze_expert_specialization_trajectory(stats)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orthogonal Expert Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Orthogonal Expert Training Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Model:</strong> {config.coupler_type}-MoE with {config.num_experts} experts</p>
            <p><strong>Configuration:</strong> {config.orthogonality_loss_type} loss, weight={config.orthogonality_loss_weight}</p>
        </div>
        
        <div class="section">
            <h2>Training Overview</h2>
            <div class="metric">
                <strong>Orthogonality Loss Enabled:</strong> {config.apply_orthogonality_loss}
            </div>
            <div class="metric">
                <strong>Loss Weight:</strong> {config.orthogonality_loss_weight}
            </div>
            <div class="metric">
                <strong>Warmup Steps:</strong> {config.orthogonality_warmup_steps}
            </div>
            <div class="metric">
                <strong>Aggregation Method:</strong> {config.orthogonality_aggregation}
            </div>
        </div>
        
        <div class="section">
            <h2>Training Curves</h2>
            <div class="plot">
                <img src="plots/training_curves.png" alt="Training Curves" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Specialization Trajectory Analysis</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
    """
    
    # Add trajectory metrics to HTML
    for category, metrics in trajectory_analysis.items():
        if metrics:
            html_content += f"<tr><td colspan='2'><strong>{category.replace('_', ' ').title()}</strong></td></tr>"
            for key, value in metrics.items():
                html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.6f}</td></tr>"
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Configuration Summary</h2>
            <table>
    """
    
    # Add configuration details
    config_items = [
        ('Experts', config.num_experts),
        ('Embedding Dimension', config.embed_dim),
        ('Model Layers', config.num_layers),
        ('Coupler Type', config.coupler_type),
        ('Orthogonality Loss Type', config.orthogonality_loss_type),
        ('Loss Weight', config.orthogonality_loss_weight),
        ('Warmup Steps', config.orthogonality_warmup_steps),
    ]
    
    for name, value in config_items:
        html_content += f"<tr><td>{name}</td><td>{value}</td></tr>"
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Orthogonality analysis report generated: {report_path}")
    return report_path

if __name__ == '__main__':
    # Example usage and testing
    print("Testing orthogonal analysis utilities...")
    
    # Create dummy expert outputs for testing
    B, L, E, D = 4, 16, 4, 64
    dummy_expert_outputs = torch.randn(B, L, E, D)
    
    # Test similarity matrix computation
    sim_matrix = compute_expert_similarity_matrix(dummy_expert_outputs, method="cosine")
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # Test orthogonality metrics
    metrics = compute_orthogonality_metrics(dummy_expert_outputs)
    print(f"Orthogonality metrics: {metrics}")
    
    # Test plotting (without saving)
    try:
        fig = plot_expert_similarity_heatmap(sim_matrix, title="Test Similarity Matrix")
        plt.close(fig)
        print("✅ Plotting functions work correctly")
    except Exception as e:
        print(f"⚠️ Plotting error (might be due to display): {e}")
    
    print("Orthogonal analysis utilities ready!")
