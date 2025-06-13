#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
components/training_controls.py

Training controls component for Streamlit dashboard.
Provides interface for starting, stopping, and monitoring training sessions.
"""

import streamlit as st
import torch
import os
import time
from datetime import datetime
from transformers import AutoTokenizer

from core.config import MoEConfig
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from utils.background_training import (
    start_background_training, stop_background_training, 
    is_training_running, get_training_error
)
from utils.state_management import (
    get_training_state, is_model_loaded, get_current_config
)

def render_training_controls(config: MoEConfig):
    """Render the training controls interface."""
    
    training_state = get_training_state()
    is_training = training_state['is_active']
    model_loaded = is_model_loaded()
    
    # Training status indicator
    if is_training:
        st.success("🚀 Training Active")
        
        # Training progress
        progress_text = training_state.get('progress_message', 'Training in progress...')
        st.info(progress_text)
        
        # Stop button
        if st.button("⏹️ Stop Training", type="secondary"):
            stop_background_training()
            st.success("Training stopped!")
            st.rerun()
    
    else:
        # Start training section
        st.subheader("🎮 Start Training")
        
        # Model status
        if model_loaded:
            st.success("✅ Model ready")
            model_info = st.session_state.get('model_info', {})
            st.write(f"**Architecture:** {model_info.get('architecture', 'N/A')}")
            st.write(f"**Parameters:** {model_info.get('parameters', 'N/A')}")
        else:
            st.warning("⚠️ No model loaded")
        
        # Configuration validation
        config_valid = validate_config_for_training(config)
        
        if config_valid['valid']:
            st.success("✅ Configuration valid")
        else:
            st.error(f"❌ Configuration issues: {config_valid['errors']}")
        
        # Training options
        with st.expander("🔧 Training Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                train_from_scratch = st.checkbox(
                    "Train from scratch",
                    value=not model_loaded,
                    help="Create a new model or use loaded model"
                )
                
                save_checkpoints = st.checkbox(
                    "Save checkpoints",
                    value=True,
                    help="Save model checkpoints during training"
                )
            
            with col2:
                auto_eval = st.checkbox(
                    "Auto evaluation",
                    value=True,
                    help="Run evaluation periodically during training"
                )
                
                live_plotting = st.checkbox(
                    "Live plotting",
                    value=True,
                    help="Update plots in real-time during training"
                )
        
        # Start training button
        start_button_disabled = not config_valid['valid']
        start_button_help = "Fix configuration issues first" if start_button_disabled else "Start training with current configuration"
        
        if st.button(
            "🚀 Start Training", 
            type="primary", 
            disabled=start_button_disabled,
            help=start_button_help
        ):
            try:
                # Initialize model if needed
                if train_from_scratch or not model_loaded:
                    success = initialize_new_model(config)
                    if not success:
                        st.error("Failed to initialize model")
                        return
                
                # Start background training
                success = start_training_session(config)
                
                if success:
                    st.success("🚀 Training started!")
                    st.info("Check the live graph for real-time progress updates.")
                    st.rerun()
                else:
                    st.error("Failed to start training")
                    
            except Exception as e:
                st.error(f"Error starting training: {str(e)}")
        
        # Show any training errors
        error = get_training_error()
        if error:
            st.error(f"Training Error: {error}")
            if st.button("🔄 Clear Error"):
                # Reset error state
                st.rerun()
    
    # Training history and quick actions
    st.divider()
    
    # Quick model creation
    if not model_loaded:
        st.subheader("⚡ Quick Start")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Create Geometric Model", help="Create a geometric training model with optimal settings"):
                create_geometric_model(config)
                st.rerun()
        
        with col2:
            if st.button("👻 Create Ghost Model", help="Create a ghost expert model for adaptive capacity"):
                create_ghost_model(config)
                st.rerun()
    
    # Recent training runs
    render_recent_runs()

def validate_config_for_training(config: MoEConfig) -> dict:
    """Validate configuration for training readiness."""
    
    errors = []
    
    # Check basic requirements
    if not config.run_name or config.run_name.strip() == "":
        errors.append("Run name is required")
    
    if config.epochs <= 0:
        errors.append("Epochs must be greater than 0")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be greater than 0")
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be greater than 0")
    
    # Check dataset configuration
    if config.dataset_source == "local_file":
        if not config.dataset_name or not os.path.exists(config.dataset_name):
            errors.append("Local dataset file not found")
    elif config.dataset_source == "huggingface":
        if not config.dataset_name:
            errors.append("HuggingFace dataset name required")
    
    # Check geometric training specific
    if config.training_mode == "geometric":
        if config.geometric.geometric_learning_rate <= 0:
            errors.append("Geometric learning rate must be greater than 0")
        if config.geometric.expert_learning_rate <= 0:
            errors.append("Expert learning rate must be greater than 0")
        if config.geometric.rotation_dimensions <= 1:
            errors.append("Rotation dimensions must be greater than 1")
    
    # Check ghost expert configuration
    if config.ghost.num_ghost_experts > 0:
        if config.ghost.ghost_learning_rate <= 0:
            errors.append("Ghost learning rate must be greater than 0")
        if config.ghost.ghost_activation_threshold <= 0:
            errors.append("Ghost activation threshold must be greater than 0")
    
    # Check model architecture
    if config.embed_dim <= 0:
        errors.append("Embedding dimension must be greater than 0")
    
    if config.num_experts <= 1:
        errors.append("Number of experts must be greater than 1")
    
    if config.embed_dim % config.num_heads != 0:
        errors.append("Embedding dimension must be divisible by number of heads")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def initialize_new_model(config: MoEConfig) -> bool:
    """Initialize a new model with the given configuration."""
    
    # Create progress placeholder for real-time updates
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Step 1: Device setup
        progress_placeholder.progress(0.1, text="🔧 Setting up compute device...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status_placeholder.info(f"💻 Using device: {device}")
        time.sleep(0.5)  # Brief pause for visibility
        
        # Step 2: Tokenizer initialization
        progress_placeholder.progress(0.2, text="📝 Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        status_placeholder.info(f"📝 Tokenizer loaded: {tokenizer.vocab_size} tokens")
        time.sleep(0.5)
        
        # Step 3: Update config
        progress_placeholder.progress(0.3, text="⚙️ Updating configuration...")
        if config.vocab_size != tokenizer.vocab_size:
            config.vocab_size = tokenizer.vocab_size
        status_placeholder.info(f"⚙️ Config updated - Vocab size: {config.vocab_size}")
        time.sleep(0.5)
        
        # Step 4: Model creation
        progress_placeholder.progress(0.5, text="🧠 Creating MoE architecture...")
        estimated_params = estimate_model_parameters(config)
        status_placeholder.info(f"🧠 Building {config.architecture_mode.upper()} model (~{estimated_params/1e6:.1f}M params)")
        
        model = MoEModel(config).to(device)
        actual_params = model.get_total_params()
        config.num_parameters = actual_params
        
        progress_placeholder.progress(0.7, text="🔧 Creating optimizers...")
        status_placeholder.info(f"✅ Model created: {actual_params/1e6:.2f}M parameters")
        time.sleep(0.5)
        
        # Step 5: Optimizer and scheduler
        optimizer = create_dynamic_optimizer(model, config)
        scheduler = PrimaryGhostLRScheduler(config, optimizer)
        
        progress_placeholder.progress(0.9, text="💾 Finalizing initialization...")
        status_placeholder.info("🔧 Optimizers and schedulers configured")
        time.sleep(0.5)
        
        # Step 6: Update session state
        progress_placeholder.progress(1.0, text="✅ Model ready!")
        
        st.session_state.update({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'tokenizer': tokenizer,
            'device': device,
            'config': config,
            'current_run_name': config.run_name,
            'model_info': {
                'architecture': config.architecture_mode,
                'parameters': f"{actual_params/1e6:.2f}M",
                'num_experts': config.num_experts,
                'num_ghost_experts': config.ghost.num_ghost_experts
            }
        })
        
        # Clear progress and show success
        progress_placeholder.empty()
        status_placeholder.success(f"✅ {config.architecture_mode.upper()} model initialized successfully!")
        st.balloons()  # Celebratory effect
        return True
        
    except Exception as e:
        # Clear progress and show error
        progress_placeholder.empty()
        status_placeholder.error(f"❌ Model initialization failed: {str(e)}")
        st.error(f"💥 Detailed error: {str(e)}")
        return False

def start_training_session(config: MoEConfig) -> bool:
    """Start a training session with the current model and config."""
    
    try:
        model = st.session_state.get('model')
        optimizer = st.session_state.get('optimizer')
        scheduler = st.session_state.get('scheduler')
        tokenizer = st.session_state.get('tokenizer')
        device = st.session_state.get('device')
        
        if not all([model, optimizer, scheduler, tokenizer, device]):
            st.error("Model components not properly initialized")
            return False
        
        # Update run name and checkpoint directory
        st.session_state['current_run_name'] = config.run_name
        
        # Start background training
        success = start_background_training(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            device=device
        )
        
        return success
        
    except Exception as e:
        st.error(f"Failed to start training: {str(e)}")
        return False

def create_geometric_model(config: MoEConfig):
    """Create a model optimized for geometric training."""
    
    # Update config for geometric training
    config.architecture_mode = "geometric"
    config.training_mode = "geometric"
    config.run_name = f"geometric_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Optimize for geometric training
    config.geometric.enabled = True
    config.geometric.geometric_learning_rate = 1e-3
    config.geometric.expert_learning_rate = 1e-4
    config.geometric.rotation_dimensions = 4
    config.geometric.lambda_cognitive_rotations = True
    
    # Initialize model
    success = initialize_new_model(config)
    if success:
        st.success("🔥 Geometric model created!")

def create_ghost_model(config: MoEConfig):
    """Create a model with ghost experts."""
    
    # Update config for ghost experts
    config.architecture_mode = "ghost"
    config.training_mode = "standard"
    config.run_name = f"ghost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Enable ghost experts
    config.ghost.num_ghost_experts = 2
    config.ghost.ghost_activation_threshold = 0.01
    config.ghost.ghost_learning_rate = 1e-4
    
    # Initialize model
    success = initialize_new_model(config)
    if success:
        st.success("👻 Ghost expert model created!")

def render_recent_runs():
    """Render recent training runs section."""
    
    st.subheader("📂 Recent Runs")
    
    # Check for recent checkpoints
    checkpoint_base = "checkpoints"
    if os.path.exists(checkpoint_base):
        recent_runs = []
        
        for run_dir in os.listdir(checkpoint_base):
            run_path = os.path.join(checkpoint_base, run_dir)
            if os.path.isdir(run_path):
                checkpoint_file = os.path.join(run_path, "checkpoint.pt")
                config_file = os.path.join(run_path, "config.json")
                
                if os.path.exists(checkpoint_file) and os.path.exists(config_file):
                    # Get modification time
                    mod_time = os.path.getmtime(checkpoint_file)
                    recent_runs.append({
                        'name': run_dir,
                        'path': run_path,
                        'modified': datetime.fromtimestamp(mod_time),
                        'checkpoint_file': checkpoint_file,
                        'config_file': config_file
                    })
        
        # Sort by modification time (most recent first)
        recent_runs.sort(key=lambda x: x['modified'], reverse=True)
        
        # Show last 3 runs
        for run in recent_runs[:3]:
            with st.expander(f"📁 {run['name']} ({run['modified'].strftime('%Y-%m-%d %H:%M')})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show run info
                    try:
                        import json
                        with open(run['config_file'], 'r') as f:
                            config_data = json.load(f)
                        
                        st.write(f"**Architecture:** {config_data.get('architecture_mode', 'N/A')}")
                        st.write(f"**Training Mode:** {config_data.get('training_mode', 'N/A')}")
                        st.write(f"**Epochs:** {config_data.get('epochs', 'N/A')}")
                        
                    except Exception as e:
                        st.error(f"Error reading config: {e}")
                
                with col2:
                    if st.button(f"📥 Load", key=f"load_recent_{run['name']}"):
                        try:
                            # Load this run's model
                            load_model_from_checkpoint(run['checkpoint_file'], run['config_file'])
                            st.success(f"Loaded {run['name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load: {str(e)}")
    else:
        st.info("No previous training runs found.")

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

def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    """Load a model from checkpoint files."""
    
    import json
    from core.training import load_checkpoint
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = MoEConfig.from_dict(config_dict)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = MoEModel(config).to(device)
    optimizer = create_dynamic_optimizer(model, config)
    scheduler = PrimaryGhostLRScheduler(config, optimizer)
    
    # Load checkpoint
    start_epoch, resume_step, best_eval_loss = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )
    
    # Update session state
    st.session_state.update({
        'config': config,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'tokenizer': tokenizer,
        'device': device,
        'current_run_name': config.run_name,
        'model_info': {
            'architecture': config.architecture_mode,
            'parameters': f"{model.get_total_params()/1e6:.2f}M",
            'num_experts': config.num_experts,
            'num_ghost_experts': config.ghost.num_ghost_experts
        }
    })
