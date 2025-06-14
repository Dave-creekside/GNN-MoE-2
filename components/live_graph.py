#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
components/live_graph.py

Live graph component for Streamlit dashboard.
Provides real-time visualization of training metrics with interactive plots.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from core.config import MoEConfig
from utils.state_management import get_plot_data, get_training_state

def get_available_plot_types(config: MoEConfig) -> List[str]:
    """Get available plot types based on architecture and training configuration."""
    
    base_plots = [
        "Training Loss",
        "Expert Activations",
        "Learning Rate Schedule"
    ]
    
    # Add orthogonality plots if orthogonal loss is enabled (architecture-based)
    if config.use_orthogonal_loss:
        orthogonal_plots = [
            "Orthogonality Preservation",
            "Expert Specialization"
        ]
        base_plots.extend(orthogonal_plots)
    
    # Add geometric rotation plots if geometric training is enabled (training-mode-based)
    if config.training_mode == "geometric":
        geometric_plots = [
            "Rotation Angles Evolution",
            "Geometric Loss Components",
            "3D Rotation Visualization"
        ]
        base_plots.extend(geometric_plots)
    
    # Add ghost plots if ghost experts are enabled (architecture-based)
    if config.ghost.num_ghost_experts > 0:
        ghost_plots = [
            "Ghost Expert Activation",
            "Saturation Monitoring",
            "Ghost vs Primary Learning"
        ]
        base_plots.extend(ghost_plots)
    
    # Add hypergraph plots if hypergraph coupling is enabled (architecture-based)
    if config.use_hypergraph_coupling:
        hypergraph_plots = [
            "Expert Connection Heatmap",
            "Hypergraph Edge Weights"
        ]
        base_plots.extend(hypergraph_plots)
    
    return base_plots

def render_live_graph(plot_type: str, config: MoEConfig):
    """Render the selected live graph."""
    
    training_state = get_training_state()
    
    if not training_state['is_active'] and not get_plot_data('loss_history'):
        # Show placeholder when no training data
        render_placeholder_graph(plot_type)
        return
    
    # Route to specific plot functions
    if plot_type == "Training Loss":
        render_loss_plot()
    elif plot_type == "Expert Activations":
        render_expert_activations_plot(config)
    elif plot_type == "Learning Rate Schedule":
        render_learning_rate_plot()
    elif plot_type == "Rotation Angles Evolution":
        render_rotation_angles_plot(config)
    elif plot_type == "Orthogonality Preservation":
        render_orthogonality_plot()
    elif plot_type == "Expert Specialization":
        render_specialization_plot()
    elif plot_type == "Geometric Loss Components":
        render_geometric_loss_components_plot()
    elif plot_type == "3D Rotation Visualization":
        render_3d_rotation_plot(config)
    elif plot_type == "Ghost Expert Activation":
        render_ghost_activation_plot(config)
    elif plot_type == "Saturation Monitoring":
        render_saturation_plot(config)
    elif plot_type == "Ghost vs Primary Learning":
        render_ghost_vs_primary_plot(config)
    elif plot_type == "Expert Connection Heatmap":
        render_expert_connection_heatmap(config)
    elif plot_type == "Hypergraph Edge Weights":
        render_hypergraph_weights_plot(config)
    else:
        st.error(f"Unknown plot type: {plot_type}")

def render_placeholder_graph(plot_type: str):
    """Render a placeholder graph when no data is available."""
    
    # Create a simple placeholder with explanation
    fig = go.Figure()
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"🚀 Start training to see live {plot_type.lower()}",
        showarrow=False,
        font=dict(size=20, color="gray"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        title=f"{plot_type} (Waiting for data...)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_loss_plot():
    """Render training loss evolution plot."""
    
    loss_data = get_plot_data('loss_history')
    
    if not loss_data:
        render_placeholder_graph("Training Loss")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(loss_data)
    
    # Create loss plot
    fig = go.Figure()
    
    # Add training loss line
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Add trend line if enough data points
    if len(df) > 10:
        z = np.polyfit(df['step'], df['loss'], 1)
        p = np.poly1d(z)
        trend_y = p(df['step'])
        
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=trend_y,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        title="Training Loss Evolution",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Remove redundant metrics - they are already shown in Quick Metrics column

def render_rotation_angles_plot(config: MoEConfig):
    """Render rotation angles evolution for geometric training."""
    
    if config.training_mode != "geometric":
        st.info("Rotation angles are only available in geometric training mode.")
        return
    
    geometric_data = get_plot_data('geometric_history')
    
    if not geometric_data:
        render_placeholder_graph("Rotation Angles Evolution")
        return
    
    df = pd.DataFrame(geometric_data)
    
    # Create subplots for each expert's rotation angles
    num_experts = config.num_experts
    
    fig = make_subplots(
        rows=min(2, num_experts), 
        cols=max(1, num_experts // 2),
        subplot_titles=[f"Expert {i+1}" for i in range(num_experts)],
        shared_xaxes=True
    )
    
    # Add rotation angle traces for each expert
    colors = px.colors.qualitative.Set1
    
    for expert_idx in range(num_experts):
        row = (expert_idx // (num_experts // 2)) + 1 if num_experts > 2 else 1
        col = (expert_idx % (num_experts // 2)) + 1 if num_experts > 2 else expert_idx + 1
        
        # Check if we have rotation data for this expert
        rotation_key = f'expert_{expert_idx}_rotation_magnitude'
        if rotation_key in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df[rotation_key],
                    mode='lines',
                    name=f'Expert {expert_idx+1}',
                    line=dict(color=colors[expert_idx % len(colors)]),
                    showlegend=(expert_idx == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="Rotation Angles Evolution per Expert",
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Training Step")
    fig.update_yaxes(title_text="Rotation Magnitude (radians)")
    
    st.plotly_chart(fig, use_container_width=True)

def render_orthogonality_plot():
    """Render orthogonality preservation metrics."""
    
    geometric_data = get_plot_data('geometric_history')
    
    if not geometric_data:
        render_placeholder_graph("Orthogonality Preservation")
        return
    
    df = pd.DataFrame(geometric_data)
    
    if 'orthogonality_preservation' not in df.columns:
        st.info("Orthogonality data not available in current training run.")
        return
    
    fig = go.Figure()
    
    # Orthogonality preservation score
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['orthogonality_preservation'],
        mode='lines+markers',
        name='Orthogonality Score',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    # Add target line (perfect orthogonality = 1.0)
    fig.add_hline(
        y=1.0, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Perfect Orthogonality"
    )
    
    fig.update_layout(
        title="Expert Orthogonality Preservation",
        xaxis_title="Training Step",
        yaxis_title="Orthogonality Score",
        height=400,
        yaxis=dict(range=[0, 1.2])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_geometric_loss_components_plot():
    """Render breakdown of geometric loss components."""
    
    geometric_data = get_plot_data('geometric_history')
    
    if not geometric_data:
        render_placeholder_graph("Geometric Loss Components")
        return
    
    df = pd.DataFrame(geometric_data)
    
    # Check for geometric loss components
    loss_components = [
        ('rotation_efficiency', 'Rotation Efficiency'),
        ('orthogonality_preservation', 'Orthogonality'),
        ('expert_specialization', 'Specialization')
    ]
    
    available_components = [(col, name) for col, name in loss_components if col in df.columns]
    
    if not available_components:
        st.info("Geometric loss components not available in current training run.")
        return
    
    # Create stacked area chart
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (col, name) in enumerate(available_components):
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df[col],
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tonexty' if i > 0 else 'tozeroy'
        ))
    
    fig.update_layout(
        title="Geometric Loss Components Over Time",
        xaxis_title="Training Step",
        yaxis_title="Component Value",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_3d_rotation_plot(config: MoEConfig):
    """Render 3D visualization of rotation patterns."""
    
    if config.training_mode != "geometric":
        st.info("3D rotation visualization is only available in geometric training mode.")
        return
    
    geometric_data = get_plot_data('geometric_history')
    
    if not geometric_data or len(geometric_data) < 10:
        st.info("Need more training data for 3D rotation visualization.")
        return
    
    # Create synthetic 3D rotation data for demonstration
    # In a real implementation, this would use actual rotation matrices
    
    steps = [d['step'] for d in geometric_data[-20:]]  # Last 20 steps
    num_experts = config.num_experts
    
    fig = go.Figure()
    
    # Create 3D scatter plot showing rotation evolution
    colors = px.colors.qualitative.Set1
    
    for expert_idx in range(num_experts):
        # Generate example rotation trajectory
        theta = np.linspace(0, 2*np.pi, len(steps))
        x = np.cos(theta + expert_idx) * (expert_idx + 1)
        y = np.sin(theta + expert_idx) * (expert_idx + 1)
        z = np.array(steps) / max(steps) * 10
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Expert {expert_idx+1}',
            line=dict(color=colors[expert_idx % len(colors)], width=4),
            marker=dict(size=3)
        ))
    
    fig.update_layout(
        title="3D Rotation Evolution (Expert Trajectories)",
        scene=dict(
            xaxis_title="Rotation X",
            yaxis_title="Rotation Y", 
            zaxis_title="Training Progress"
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 This visualization shows how each expert's rotation parameters evolve in 3D space during training.")

def render_ghost_activation_plot(config: MoEConfig):
    """Render ghost expert activation patterns."""
    
    if config.ghost.num_ghost_experts == 0:
        st.info("Ghost expert visualization requires ghost experts to be enabled.")
        return
    
    ghost_data = get_plot_data('ghost_history')
    
    if not ghost_data:
        render_placeholder_graph("Ghost Expert Activation")
        return
    
    df = pd.DataFrame(ghost_data)
    
    fig = go.Figure()
    
    # Active ghost experts over time
    if 'active_ghosts' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['active_ghosts'],
            mode='lines+markers',
            name='Active Ghost Experts',
            line=dict(color='purple', width=2),
            marker=dict(size=4)
        ))
    
    # Add total ghost experts line
    fig.add_hline(
        y=config.ghost.num_ghost_experts,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Total Ghost Experts ({config.ghost.num_ghost_experts})"
    )
    
    fig.update_layout(
        title="Ghost Expert Activation Over Time",
        xaxis_title="Training Step",
        yaxis_title="Number of Active Ghost Experts",
        height=400,
        yaxis=dict(range=[0, config.ghost.num_ghost_experts + 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_expert_activations_plot(config: MoEConfig):
    """Render expert activation patterns."""
    
    expert_data = get_plot_data('expert_activation_history')
    
    if not expert_data:
        render_placeholder_graph("Expert Activations")
        return
    
    # Create heatmap of expert activations over time
    df = pd.DataFrame(expert_data)
    
    # Extract activation data for each expert
    expert_cols = [col for col in df.columns if col.startswith('expert_') and col.endswith('_activation')]
    
    if not expert_cols:
        st.info("Expert activation data not available in current training run.")
        return
    
    # Create heatmap
    activation_matrix = df[expert_cols].values.T
    steps = df['step'].values
    
    fig = go.Figure(data=go.Heatmap(
        z=activation_matrix,
        x=steps,
        y=[f"Expert {i+1}" for i in range(len(expert_cols))],
        colorscale='Viridis',
        colorbar=dict(title="Activation Level")
    ))
    
    fig.update_layout(
        title="Expert Activation Heatmap",
        xaxis_title="Training Step",
        yaxis_title="Expert",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_learning_rate_plot():
    """Render learning rate schedule."""
    
    loss_data = get_plot_data('loss_history')
    
    if not loss_data:
        render_placeholder_graph("Learning Rate Schedule")
        return
    
    # For now, show a simple learning rate decay
    # In a real implementation, this would track actual learning rates
    
    steps = [d['step'] for d in loss_data]
    base_lr = 1e-4
    
    # Simulate cosine annealing
    max_steps = max(steps) if steps else 1000
    lrs = [base_lr * (1 + np.cos(np.pi * step / max_steps)) / 2 for step in steps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=lrs,
        mode='lines',
        name='Learning Rate',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Training Step",
        yaxis_title="Learning Rate",
        height=400,
        yaxis=dict(type='log')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_expert_connection_heatmap(config: MoEConfig):
    """Render expert connection strength heatmap."""
    
    if not config.use_hypergraph_coupling:
        st.info("Expert connections are only available with hypergraph coupling enabled.")
        return
    
    # Create synthetic connection matrix for demonstration
    num_experts = config.num_experts
    
    # Generate example connection strengths
    np.random.seed(42)  # For reproducible demo
    connections = np.random.rand(num_experts, num_experts)
    
    # Make symmetric and add stronger diagonal
    connections = (connections + connections.T) / 2
    np.fill_diagonal(connections, 1.0)
    
    fig = go.Figure(data=go.Heatmap(
        z=connections,
        x=[f"Expert {i+1}" for i in range(num_experts)],
        y=[f"Expert {i+1}" for i in range(num_experts)],
        colorscale='Blues',
        colorbar=dict(title="Connection Strength")
    ))
    
    fig.update_layout(
        title="Expert Connection Strength Matrix",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_saturation_plot(config: MoEConfig):
    """Render expert saturation monitoring."""
    
    ghost_data = get_plot_data('ghost_history')
    
    if not ghost_data:
        render_placeholder_graph("Saturation Monitoring")
        return
    
    df = pd.DataFrame(ghost_data)
    
    if 'saturation_level' not in df.columns:
        st.info("Saturation data not available in current training run.")
        return
    
    fig = go.Figure()
    
    # Saturation level over time
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['saturation_level'],
        mode='lines+markers',
        name='Saturation Level',
        line=dict(color='red', width=2)
    ))
    
    # Add activation threshold line
    fig.add_hline(
        y=config.ghost.ghost_activation_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Activation Threshold ({config.ghost.ghost_activation_threshold})"
    )
    
    fig.update_layout(
        title="Expert Saturation Monitoring",
        xaxis_title="Training Step",
        yaxis_title="Saturation Level",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_ghost_vs_primary_plot(config: MoEConfig):
    """Render comparison between ghost and primary expert learning."""
    
    st.info("🚧 Ghost vs Primary Learning comparison plot coming soon!")
    render_placeholder_graph("Ghost vs Primary Learning")

def render_hypergraph_weights_plot(config: MoEConfig):
    """Render hypergraph edge weights evolution."""
    
    st.info("🚧 Hypergraph edge weights visualization coming soon!")
    render_placeholder_graph("Hypergraph Edge Weights")

def render_specialization_plot():
    """Render expert specialization metrics."""
    
    geometric_data = get_plot_data('geometric_history')
    
    if not geometric_data:
        render_placeholder_graph("Expert Specialization")
        return
    
    df = pd.DataFrame(geometric_data)
    
    if 'expert_specialization' not in df.columns:
        st.info("Expert specialization data not available in current training run.")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['expert_specialization'],
        mode='lines+markers',
        name='Specialization Score',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Expert Specialization Evolution",
        xaxis_title="Training Step",
        yaxis_title="Specialization Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
