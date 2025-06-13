#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
components/config_panel.py

Configuration panel component for Streamlit dashboard.
Provides interactive UI for configuring MoE model parameters.
"""

import streamlit as st
import os
from typing import Dict, Any

from core.config import MoEConfig, HGNNParams, GhostParams, GeometricTrainingConfig
from utils.state_management import (
    get_current_config, update_config, get_config_preset, save_config_preset
)

def render_config_panel() -> MoEConfig:
    """Render the configuration panel in the sidebar."""
    
    # Get current config
    config = get_current_config()
    
    # Configuration presets
    st.subheader("📋 Configuration Presets")
    
    col1, col2 = st.columns(2)
    with col1:
        preset_options = ["Custom", "Geometric Lambda", "Ghost Expert Test", "Basic Comparison"]
        selected_preset = st.selectbox(
            "Load Preset:",
            preset_options,
            index=0,
            help="Load a predefined configuration"
        )
    
    with col2:
        if st.button("📥 Load"):
            if selected_preset != "Custom":
                preset_map = {
                    "Geometric Lambda": "geometric_lambda",
                    "Ghost Expert Test": "ghost_expert_test", 
                    "Basic Comparison": "basic_comparison"
                }
                preset_config = get_config_preset(preset_map[selected_preset])
                if preset_config:
                    update_config(preset_config)
                    st.success(f"Loaded {selected_preset} preset!")
                    st.rerun()
    
    st.divider()
    
    # Core Architecture Settings
    st.subheader("🏗️ Architecture")
    
    # Architecture mode selection
    architecture_options = ["gnn", "hgnn", "orthogonal", "ghost", "geometric"]
    architecture_descriptions = {
        "gnn": "Basic GNN coupling between experts",
        "hgnn": "Hypergraph coupling for complex relationships",
        "orthogonal": "HGNN + orthogonality constraints",
        "ghost": "All features + adaptive ghost experts",
        "geometric": "Revolutionary geometric constrained learning"
    }
    
    selected_architecture = st.selectbox(
        "Architecture Mode:",
        architecture_options,
        index=architecture_options.index(config.architecture_mode),
        format_func=lambda x: f"{x.upper()} - {architecture_descriptions[x]}",
        help="Select the MoE architecture variant"
    )
    
    if selected_architecture != config.architecture_mode:
        config.architecture_mode = selected_architecture
        # Trigger __post_init__ to update feature flags
        config.__post_init__()
    
    # Training mode (geometric vs standard)
    training_options = ["standard", "geometric"]
    selected_training_mode = st.selectbox(
        "Training Mode:",
        training_options,
        index=training_options.index(config.training_mode),
        help="Choose between standard training and geometric constrained learning"
    )
    config.training_mode = selected_training_mode
    
    st.divider()
    
    # Core Model Parameters
    st.subheader("🔧 Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.run_name = st.text_input(
            "Run Name:",
            value=config.run_name,
            help="Name for this experiment run"
        )
        
        config.num_experts = st.slider(
            "Number of Experts:",
            min_value=2,
            max_value=16,
            value=config.num_experts,
            help="Number of expert networks in each MoE layer"
        )
        
        config.embed_dim = st.slider(
            "Embedding Dimension:",
            min_value=32,
            max_value=512,
            value=config.embed_dim,
            step=32,
            help="Dimensionality of token embeddings"
        )
    
    with col2:
        config.batch_size = st.slider(
            "Batch Size:",
            min_value=1,
            max_value=32,
            value=config.batch_size,
            help="Training batch size"
        )
        
        config.epochs = st.slider(
            "Epochs:",
            min_value=1,
            max_value=50,
            value=config.epochs,
            help="Number of training epochs"
        )
        
        config.learning_rate = st.number_input(
            "Learning Rate:",
            min_value=1e-6,
            max_value=1e-2,
            value=config.learning_rate,
            format="%.6f",
            help="Base learning rate for training"
        )
    
    # Architecture-specific settings
    if config.architecture_mode in ["hgnn", "orthogonal", "ghost"] or config.use_hypergraph_coupling:
        st.divider()
        st.subheader("🕸️ Hypergraph Settings")
        
        config.hgnn.num_layers = st.slider(
            "HGNN Layers:",
            min_value=1,
            max_value=5,
            value=config.hgnn.num_layers,
            help="Number of hypergraph neural network layers"
        )
        
        config.hgnn.strategy = st.selectbox(
            "Hypergraph Strategy:",
            ["all_pairs", "all_triplets", "all"],
            index=["all_pairs", "all_triplets", "all"].index(config.hgnn.strategy),
            help="Strategy for creating hypergraph connections"
        )
        
        config.hgnn.learnable_edge_weights = st.checkbox(
            "Learnable Edge Weights",
            value=config.hgnn.learnable_edge_weights,
            help="Allow hypergraph edge weights to be learned during training"
        )
    
    # Ghost expert settings
    if config.architecture_mode == "ghost":
        st.divider()
        st.subheader("👻 Ghost Expert Settings")
        
        config.ghost.num_ghost_experts = st.slider(
            "Number of Ghost Experts:",
            min_value=0,
            max_value=8,
            value=config.ghost.num_ghost_experts,
            help="Number of dormant ghost experts for adaptive capacity"
        )
        
        if config.ghost.num_ghost_experts > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                config.ghost.ghost_activation_threshold = st.number_input(
                    "Activation Threshold:",
                    min_value=0.001,
                    max_value=0.1,
                    value=config.ghost.ghost_activation_threshold,
                    format="%.3f",
                    help="Saturation threshold for ghost activation"
                )
                
                config.ghost.ghost_learning_rate = st.number_input(
                    "Ghost Learning Rate:",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=config.ghost.ghost_learning_rate,
                    format="%.6f",
                    help="Learning rate for ghost experts"
                )
            
            with col2:
                config.ghost.ghost_activation_schedule = st.selectbox(
                    "Activation Schedule:",
                    ["gradual", "binary", "selective"],
                    index=["gradual", "binary", "selective"].index(config.ghost.ghost_activation_schedule),
                    help="How ghost experts are activated over time"
                )
                
                config.ghost.saturation_monitoring_window = st.number_input(
                    "Monitoring Window:",
                    min_value=10,
                    max_value=1000,
                    value=config.ghost.saturation_monitoring_window,
                    help="Steps to average for saturation monitoring"
                )
    
    # Geometric training settings
    if config.training_mode == "geometric" or config.architecture_mode == "geometric":
        st.divider()
        st.subheader("🔄 Geometric Constrained Learning")
        
        config.geometric.enabled = True
        
        col1, col2 = st.columns(2)
        
        with col1:
            config.geometric.geometric_learning_rate = st.number_input(
                "Geometric Learning Rate:",
                min_value=1e-6,
                max_value=1e-1,
                value=config.geometric.geometric_learning_rate,
                format="%.6f",
                help="Learning rate for rotation parameters (typically higher)"
            )
            
            config.geometric.expert_learning_rate = st.number_input(
                "Expert Learning Rate:",
                min_value=1e-6,
                max_value=1e-2,
                value=config.geometric.expert_learning_rate,
                format="%.6f",
                help="Learning rate for expert parameters (typically lower)"
            )
            
            config.geometric.rotation_dimensions = st.slider(
                "Rotation Dimensions:",
                min_value=2,
                max_value=16,
                value=config.geometric.rotation_dimensions,
                help="Dimensionality of rotation transformations"
            )
        
        with col2:
            st.write("**Loss Weights:**")
            config.geometric.orthogonality_weight = st.slider(
                "Orthogonality:",
                min_value=0.0,
                max_value=2.0,
                value=config.geometric.orthogonality_weight,
                step=0.1,
                help="Weight for orthogonality preservation loss"
            )
            
            config.geometric.rotation_efficiency_weight = st.slider(
                "Rotation Efficiency:",
                min_value=0.0,
                max_value=2.0,
                value=config.geometric.rotation_efficiency_weight,
                step=0.1,
                help="Weight for rotation efficiency loss"
            )
            
            config.geometric.specialization_weight = st.slider(
                "Specialization:",
                min_value=0.0,
                max_value=2.0,
                value=config.geometric.specialization_weight,
                step=0.1,
                help="Weight for expert specialization loss"
            )
        
        # Lambda calculus specific settings
        st.write("**Lambda Calculus Optimization:**")
        config.geometric.lambda_cognitive_rotations = st.checkbox(
            "Enable Lambda Cognitive Rotations",
            value=config.geometric.lambda_cognitive_rotations,
            help="Use specialized rotations for lambda calculus reasoning"
        )
        
        if config.geometric.lambda_cognitive_rotations:
            config.geometric.lambda_rotation_scheduling = st.selectbox(
                "Lambda Rotation Scheduling:",
                ["curriculum", "adaptive", "fixed"],
                index=["curriculum", "adaptive", "fixed"].index(config.geometric.lambda_rotation_scheduling),
                help="How lambda-specific rotations are scheduled during training"
            )
    
    st.divider()
    
    # Dataset Configuration
    st.subheader("📁 Dataset")
    
    dataset_source = st.selectbox(
        "Dataset Source:",
        ["huggingface", "local_file"],
        index=["huggingface", "local_file"].index(config.dataset_source),
        help="Choose between HuggingFace datasets or local files"
    )
    config.dataset_source = dataset_source
    
    if dataset_source == "huggingface":
        config.dataset_name = st.text_input(
            "Dataset Name:",
            value=config.dataset_name,
            placeholder="e.g., openai/gsm8k, wikitext/wikitext-103-v1",
            help="HuggingFace dataset path"
        )
        
        config.dataset_config_name = st.text_input(
            "Config Name (optional):",
            value=config.dataset_config_name,
            placeholder="e.g., main, wikitext-103-v1",
            help="Specific configuration for the dataset"
        )
    
    else:  # local_file
        # Show available local files
        data_dir = "data"
        if os.path.exists(data_dir):
            local_files = [f for f in os.listdir(data_dir) 
                          if f.endswith(('.json', '.jsonl', '.txt'))]
            if local_files:
                selected_file = st.selectbox(
                    "Select Local File:",
                    [""] + local_files,
                    index=local_files.index(os.path.basename(config.dataset_name)) + 1 
                          if os.path.basename(config.dataset_name) in local_files else 0,
                    help="Choose from available local dataset files"
                )
                if selected_file:
                    config.dataset_name = os.path.join(data_dir, selected_file)
            else:
                st.info("No local dataset files found in data/ directory")
        
        # Manual path input as fallback
        config.dataset_name = st.text_input(
            "Dataset Path:",
            value=config.dataset_name,
            placeholder="data/your_dataset.json",
            help="Path to your local dataset file"
        )
    
    # Dataset size controls
    col1, col2 = st.columns(2)
    with col1:
        config.num_train_samples = st.number_input(
            "Training Samples:",
            min_value=-1,
            max_value=100000,
            value=config.num_train_samples,
            help="Number of training samples (-1 for all)"
        )
    with col2:
        config.num_eval_samples = st.number_input(
            "Evaluation Samples:",
            min_value=-1,
            max_value=10000,
            value=config.num_eval_samples,
            help="Number of evaluation samples (-1 for all)"
        )
    
    st.divider()
    
    # Advanced Settings
    st.subheader("⚙️ Advanced Settings")
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns(2)
    with col1:
        config.num_layers = st.slider(
            "Transformer Layers:",
            min_value=1,
            max_value=12,
            value=config.num_layers,
            help="Number of transformer layers"
        )
        
        config.num_heads = st.slider(
            "Attention Heads:",
            min_value=1,
            max_value=16,
            value=config.num_heads,
            help="Number of attention heads"
        )
    
    with col2:
        config.max_seq_length = st.slider(
            "Max Sequence Length:",
            min_value=64,
            max_value=2048,
            value=config.max_seq_length,
            step=64,
            help="Maximum input sequence length"
        )
        
        config.dropout_rate = st.slider(
            "Dropout Rate:",
            min_value=0.0,
            max_value=0.5,
            value=config.dropout_rate,
            step=0.05,
            help="Dropout rate for regularization"
        )
    
    st.subheader("Training")
    config.max_batches_per_epoch = st.number_input(
        "Max Batches per Epoch:",
        min_value=-1,
        max_value=10000,
        value=config.max_batches_per_epoch,
        help="Limit batches per epoch (-1 for all)"
    )
    
    config.eval_every = st.number_input(
        "Evaluate Every N Steps:",
        min_value=10,
        max_value=1000,
        value=config.eval_every,
        help="Frequency of evaluation during training"
    )
    
    config.log_every = st.number_input(
        "Log Every N Steps:",
        min_value=1,
        max_value=100,
        value=config.log_every,
        help="Frequency of progress logging"
    )
    
    # Save current config
    update_config(config)
    
    # Config validation
    if st.button("✅ Validate Configuration"):
        try:
            # Test config creation
            config.__post_init__()
            st.success("✅ Configuration is valid!")
            
            # Show key info
            st.info(f"""
            **Configuration Summary:**
            - Architecture: {config.architecture_mode.upper()}
            - Training Mode: {config.training_mode.upper()}
            - Experts: {config.num_experts} primary + {config.ghost.num_ghost_experts if config.ghost.num_ghost_experts > 0 else 0} ghost
            - Parameters: ~{estimate_model_parameters(config)/1e6:.1f}M
            - Dataset: {config.dataset_source} → {config.dataset_name}
            """)
            
        except Exception as e:
            st.error(f"❌ Configuration error: {str(e)}")
    
    return config

def estimate_model_parameters(config: MoEConfig) -> int:
    """Estimate the number of model parameters based on configuration."""
    
    # Basic transformer parameters
    vocab_size = config.vocab_size
    embed_dim = config.embed_dim
    num_layers = config.num_layers
    num_heads = config.num_heads
    max_seq_len = config.max_seq_length
    
    # Embedding layers
    token_embed = vocab_size * embed_dim
    pos_embed = max_seq_len * embed_dim
    
    # Transformer layers (rough estimate)
    attention_params = embed_dim * embed_dim * 4 * num_heads  # Q, K, V, O projections
    ffn_params = embed_dim * embed_dim * 4 * 2  # Two linear layers with 4x expansion
    layer_params = (attention_params + ffn_params) * num_layers
    
    # MoE-specific parameters
    expert_params = 0
    num_experts = config.num_experts + config.ghost.num_ghost_experts
    
    # Each expert is essentially an FFN
    expert_ffn_params = embed_dim * embed_dim * 4 * 2
    expert_params = expert_ffn_params * num_experts * num_layers
    
    # HGNN parameters (if enabled)
    hgnn_params = 0
    if config.use_hypergraph_coupling:
        # Rough estimate for hypergraph parameters
        hgnn_params = embed_dim * embed_dim * config.hgnn.num_layers
    
    # Geometric parameters (if enabled)
    geometric_params = 0
    if config.training_mode == "geometric":
        # Rotation parameters
        geometric_params = config.geometric.rotation_dimensions * num_experts * num_layers
    
    # Output layer
    output_params = embed_dim * vocab_size
    
    total_params = (token_embed + pos_embed + layer_params + 
                   expert_params + hgnn_params + geometric_params + output_params)
    
    return int(total_params)
