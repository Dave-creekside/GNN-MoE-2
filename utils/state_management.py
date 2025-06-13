#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/state_management.py

Session state management for Streamlit dashboard.
Handles persistent state across page interactions and training runs.
"""

import streamlit as st
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

from core.config import MoEConfig

def initialize_session_state():
    """Initialize all session state variables with defaults."""
    
    # Configuration state
    if 'config' not in st.session_state:
        st.session_state.config = MoEConfig()
    
    # Model and training state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'device' not in st.session_state:
        st.session_state.device = None
    
    # Training state
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = None
    if 'current_run_name' not in st.session_state:
        st.session_state.current_run_name = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    
    # Training metrics and monitoring
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {
            'is_active': False,
            'current_epoch': 0,
            'current_step': 0,
            'total_epochs': 0,
            'current_loss': None,
            'loss_delta': None,
            'geometric_metrics': {},
            'ghost_metrics': {},
            'start_time': None,
            'last_update': None
        }
    
    # Live plotting data
    if 'plot_data' not in st.session_state:
        st.session_state.plot_data = {
            'loss_history': [],
            'geometric_history': [],
            'ghost_history': [],
            'expert_activation_history': []
        }
    
    # Analysis data cache
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    # Configuration history for quick presets
    if 'config_presets' not in st.session_state:
        st.session_state.config_presets = {
            'geometric_lambda': create_geometric_lambda_preset(),
            'ghost_expert_test': create_ghost_expert_preset(),
            'basic_comparison': create_basic_comparison_preset()
        }

def create_geometric_lambda_preset() -> MoEConfig:
    """Create a preset configuration for geometric lambda calculus research."""
    config = MoEConfig()
    config.run_name = "geometric_lambda_research"
    config.architecture_mode = "geometric"
    config.training_mode = "geometric"
    config.num_experts = 4
    config.embed_dim = 128
    config.epochs = 10
    config.batch_size = 8
    
    # Geometric settings
    config.geometric.enabled = True
    config.geometric.geometric_learning_rate = 1e-3
    config.geometric.expert_learning_rate = 1e-4
    config.geometric.rotation_dimensions = 4
    config.geometric.lambda_cognitive_rotations = True
    config.geometric.lambda_rotation_scheduling = "curriculum"
    
    # Dataset
    config.dataset_source = "huggingface"
    config.dataset_name = "openai/gsm8k"  # Use a real dataset for now
    config.dataset_config_name = ""  # Use default config
    
    return config

def create_ghost_expert_preset() -> MoEConfig:
    """Create a preset configuration for ghost expert testing."""
    config = MoEConfig()
    config.run_name = "ghost_expert_test"
    config.architecture_mode = "ghost"
    config.training_mode = "standard"
    config.num_experts = 4
    config.ghost.num_ghost_experts = 2
    config.ghost.ghost_activation_threshold = 0.01
    config.ghost.ghost_learning_rate = 1e-4
    config.embed_dim = 128
    config.epochs = 5
    config.batch_size = 8
    
    return config

def create_basic_comparison_preset() -> MoEConfig:
    """Create a basic configuration for comparison experiments."""
    config = MoEConfig()
    config.run_name = "basic_comparison"
    config.architecture_mode = "hgnn"
    config.training_mode = "standard"
    config.num_experts = 4
    config.embed_dim = 64
    config.epochs = 3
    config.batch_size = 4
    
    return config

def get_training_state() -> Dict[str, Any]:
    """Get current training state from session."""
    return st.session_state.training_metrics

def update_training_state(updates: Dict[str, Any]):
    """Update training state with new metrics."""
    st.session_state.training_metrics.update(updates)
    st.session_state.training_metrics['last_update'] = datetime.now()

def start_training_session(config: MoEConfig):
    """Initialize a new training session."""
    st.session_state.training_metrics.update({
        'is_active': True,
        'current_epoch': 0,
        'current_step': 0,
        'total_epochs': config.epochs,
        'current_loss': None,
        'loss_delta': None,
        'start_time': datetime.now(),
        'geometric_metrics': {},
        'ghost_metrics': {}
    })
    
    # Clear previous plot data
    st.session_state.plot_data = {
        'loss_history': [],
        'geometric_history': [],
        'ghost_history': [],
        'expert_activation_history': []
    }

def stop_training_session():
    """Stop the current training session."""
    st.session_state.training_metrics.update({
        'is_active': False,
        'last_update': datetime.now()
    })

def add_training_datapoint(step: int, epoch: int, loss: float, 
                          geometric_metrics: Optional[Dict] = None,
                          ghost_metrics: Optional[Dict] = None,
                          expert_activations: Optional[Dict] = None):
    """Add a new datapoint to the training history."""
    
    # Update current state
    previous_loss = st.session_state.training_metrics.get('current_loss')
    loss_delta = (loss - previous_loss) if previous_loss is not None else None
    
    update_training_state({
        'current_step': step,
        'current_epoch': epoch,
        'current_loss': loss,
        'loss_delta': loss_delta,
        'geometric_metrics': geometric_metrics or {},
        'ghost_metrics': ghost_metrics or {},
    })
    
    # Add to plot data history
    timestamp = datetime.now()
    
    # Loss history
    st.session_state.plot_data['loss_history'].append({
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'timestamp': timestamp
    })
    
    # Geometric history
    if geometric_metrics:
        st.session_state.plot_data['geometric_history'].append({
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            **geometric_metrics
        })
    
    # Ghost history
    if ghost_metrics:
        st.session_state.plot_data['ghost_history'].append({
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            **ghost_metrics
        })
    
    # Expert activations
    if expert_activations:
        st.session_state.plot_data['expert_activation_history'].append({
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            **expert_activations
        })
    
    # Keep only last 1000 datapoints to prevent memory issues
    for key in st.session_state.plot_data:
        if len(st.session_state.plot_data[key]) > 1000:
            st.session_state.plot_data[key] = st.session_state.plot_data[key][-1000:]

def get_plot_data(data_type: str) -> list:
    """Get plotting data for a specific type."""
    return st.session_state.plot_data.get(data_type, [])

def get_config_preset(preset_name: str) -> Optional[MoEConfig]:
    """Get a configuration preset by name."""
    return st.session_state.config_presets.get(preset_name)

def save_config_preset(name: str, config: MoEConfig):
    """Save a configuration as a preset."""
    st.session_state.config_presets[name] = config

def clear_session_data():
    """Clear all session data (useful for reset)."""
    keys_to_keep = ['config_presets']  # Keep presets
    keys_to_clear = [k for k in st.session_state.keys() if k not in keys_to_keep]
    
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()

def get_training_summary() -> Dict[str, Any]:
    """Get a summary of the current training session."""
    state = get_training_state()
    
    if not state['is_active']:
        return {'status': 'inactive'}
    
    duration = None
    if state.get('start_time'):
        duration = datetime.now() - state['start_time']
    
    return {
        'status': 'active',
        'duration': duration,
        'progress': state.get('current_epoch', 0) / max(state.get('total_epochs', 1), 1),
        'current_epoch': state.get('current_epoch', 0),
        'total_epochs': state.get('total_epochs', 0),
        'current_step': state.get('current_step', 0),
        'current_loss': state.get('current_loss'),
        'loss_trend': 'improving' if (state.get('loss_delta', 0) < 0) else 'worsening' if (state.get('loss_delta', 0) > 0) else 'stable'
    }

def is_model_loaded() -> bool:
    """Check if a model is currently loaded."""
    return st.session_state.get('model') is not None

def is_training_active() -> bool:
    """Check if training is currently active."""
    return st.session_state.training_metrics.get('is_active', False)

def get_current_config() -> MoEConfig:
    """Get the current configuration."""
    return st.session_state.config

def update_config(config: MoEConfig):
    """Update the current configuration."""
    st.session_state.config = config
