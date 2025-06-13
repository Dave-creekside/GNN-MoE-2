#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils

Utility modules for Streamlit dashboard.
"""

from .state_management import (
    initialize_session_state, get_training_state, update_training_state,
    start_training_session, stop_training_session, add_training_datapoint,
    get_plot_data, get_config_preset, save_config_preset,
    clear_session_data, get_training_summary, is_model_loaded, 
    is_training_active, get_current_config, update_config
)

from .background_training import (
    BackgroundTrainingManager, start_background_training, 
    stop_background_training, is_training_running, get_training_error
)

__all__ = [
    # State management
    'initialize_session_state', 'get_training_state', 'update_training_state',
    'start_training_session', 'stop_training_session', 'add_training_datapoint',
    'get_plot_data', 'get_config_preset', 'save_config_preset',
    'clear_session_data', 'get_training_summary', 'is_model_loaded', 
    'is_training_active', 'get_current_config', 'update_config',
    
    # Background training
    'BackgroundTrainingManager', 'start_background_training', 
    'stop_background_training', 'is_training_running', 'get_training_error'
]
