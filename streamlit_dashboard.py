#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_dashboard.py

Interactive Streamlit dashboard for the MoE Research Hub.
Provides visual interface for configuration, training monitoring, and analysis.
"""

import streamlit as st
import torch
import os
import json
import threading
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Import core modules
from core.config import MoEConfig, HGNNParams, GhostParams, GeometricTrainingConfig
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.training import controller_training_loop, load_checkpoint
from core.data import load_data_with_preprocessing
from core.inference import generate_text
from core.analysis import load_log_data

# Dashboard components
from components.config_panel import render_config_panel
from components.live_graph import render_live_graph, get_available_plot_types
from components.training_controls import render_training_controls
from utils.background_training import BackgroundTrainingManager
from utils.state_management import initialize_session_state, get_training_state, get_current_config

# Page configuration
st.set_page_config(
    page_title="🧠 MoE Research Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better layout
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 1rem;
    }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .training-status {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Initialize session state
    initialize_session_state()
    
    
    # Main layout: sidebar + main content
    with st.sidebar:
        # Load config first (but don't render the full panel yet)
        config = get_current_config()
        
        # Training controls at the top - always visible
        st.header("🎮 Training Controls")
        render_training_controls(config)
        
        st.divider()
        
        # Configuration - collapsible
        with st.expander("⚙️ Configuration", expanded=False):
            config = render_config_panel()
        
        # Training status - collapsible
        training_state = get_training_state()
        if training_state['is_active']:
            with st.expander("📊 Training Status", expanded=True):
                st.markdown('<div class="training-status">', unsafe_allow_html=True)
                st.success("🚀 Training Active")
                st.write(f"**Epoch:** {training_state.get('current_epoch', 'N/A')}")
                st.write(f"**Step:** {training_state.get('current_step', 'N/A')}")
                st.write(f"**Loss:** {training_state.get('current_loss', 'N/A'):.4f}" if training_state.get('current_loss') else "**Loss:** N/A")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.expander("📊 Training Status", expanded=False):
                st.info("⏸️ No active training")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Visualization")
        
        # Plot type selector
        plot_types = get_available_plot_types(config)
        selected_plot = st.selectbox(
            "Select visualization:",
            plot_types,
            index=0,
            help="Choose the type of real-time plot to display"
        )
        
        # Live graph area
        graph_placeholder = st.empty()
        
        # Render the selected live graph
        with graph_placeholder.container():
            render_live_graph(selected_plot, config)
    
    with col2:
        # Four clean progress bars only - no header, no cards, no other content
        training_state = get_training_state()
        
        # 1. Epoch Progress
        current_epoch = training_state.get('current_epoch', 0)
        total_epochs = training_state.get('total_epochs', 1)
        epoch_progress = min(current_epoch / max(total_epochs, 1), 1.0)
        st.progress(
            epoch_progress,
            text=f"Epochs: {current_epoch}/{total_epochs} ({epoch_progress*100:.1f}%)"
        )
        
        # 2. Step Progress (total across all epochs)
        current_step = training_state.get('current_step', 0)
        # Calculate total estimated steps (rough estimate: current progress * total epochs / current epoch)
        if current_epoch > 0 and current_step > 0:
            estimated_steps_per_epoch = max(1, current_step // max(current_epoch, 1))
            total_estimated_steps = estimated_steps_per_epoch * total_epochs
            step_progress = min(current_step / max(total_estimated_steps, 1), 1.0)
        else:
            total_estimated_steps = 100 * total_epochs  # Default estimate
            step_progress = 0.0
        st.progress(
            step_progress,
            text=f"Steps: {current_step}/{total_estimated_steps if current_step > 0 else '?'} ({step_progress*100:.1f}%)"
        )
        
        # 3. Active Ghost Experts
        ghost_metrics = training_state.get('ghost_metrics', {})
        active_ghosts = ghost_metrics.get('active_ghosts', 0)
        total_ghosts = config.ghost.num_ghost_experts if config else 0
        if total_ghosts > 0:
            ghost_progress = active_ghosts / total_ghosts
            st.progress(
                ghost_progress,
                text=f"Ghost Experts: {active_ghosts}/{total_ghosts} ({ghost_progress*100:.1f}%)"
            )
        else:
            st.progress(0.0, text="Ghost Experts: 0/0 (0.0%)")
        
        # 4. Loss Reduction Percentage
        current_loss = training_state.get('current_loss')
        loss_history = st.session_state.plot_data.get('loss_history', [])
        if current_loss is not None and len(loss_history) > 1:
            initial_loss = loss_history[0]['loss']
            if initial_loss > current_loss:
                loss_reduction_pct = ((initial_loss - current_loss) / initial_loss) * 100
                # Cap at 100% for display
                display_progress = min(loss_reduction_pct / 100, 1.0)
                st.progress(
                    display_progress,
                    text=f"Loss Reduction: {loss_reduction_pct:.1f}% (from {initial_loss:.3f})"
                )
            else:
                st.progress(0.0, text=f"Loss: {current_loss:.3f} (no reduction yet)")
        else:
            st.progress(0.0, text="Loss: Waiting for data...")
    
    # Tabs for additional functionality
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🎯 Inference", "💾 Models", "📁 Datasets"])
    
    with tab1:
        st.header("Training Analysis")
        
        # Load and display analysis plots
        if st.session_state.get('current_run_name'):
            checkpoint_dir = f"checkpoints/{st.session_state['current_run_name']}"
            log_path = os.path.join(checkpoint_dir, "training_log.json")
            
            if os.path.exists(log_path):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Refresh Analysis"):
                        st.session_state['analysis_data'] = load_log_data(log_path)
                
                with col2:
                    if st.button("📈 Generate Full Report"):
                        from core.analysis import run_analysis
                        run_analysis(log_path)
                        st.success("Analysis plots generated!")
                
                # Display key plots if data is available
                if 'analysis_data' in st.session_state:
                    df = st.session_state['analysis_data']
                    
                    # Loss evolution
                    st.subheader("Loss Evolution")
                    fig = px.line(df, x='step', y=['train_loss', 'eval_loss'], 
                                title="Training vs Validation Loss")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Geometric metrics (if applicable)
                    if 'rotation_efficiency' in df.columns:
                        st.subheader("Geometric Training Metrics")
                        fig = make_subplots(rows=2, cols=2, 
                                          subplot_titles=['Rotation Efficiency', 'Orthogonality Preservation',
                                                        'Expert Specialization', 'Rotation Magnitude'])
                        
                        fig.add_trace(go.Scatter(x=df['step'], y=df['rotation_efficiency'], name='Rotation Efficiency'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df['step'], y=df['orthogonality_preservation'], name='Orthogonality'), row=1, col=2)
                        fig.add_trace(go.Scatter(x=df['step'], y=df['expert_specialization'], name='Specialization'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=df['step'], y=df['avg_rotation_magnitude'], name='Rotation Magnitude'), row=2, col=2)
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training data available. Start training to generate analysis.")
        else:
            st.info("No active training run. Configure and start training to see analysis.")
    
    with tab2:
        st.header("Interactive Inference")
        
        if st.session_state.get('model') and st.session_state.get('tokenizer'):
            prompt = st.text_area("Enter your prompt:", 
                                value="Question: What is the result of ((\u03bbx.(\u03bby.(x y))) a) b?\nReasoning:",
                                height=100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                max_length = st.slider("Max new tokens", 10, 200, 50)
            with col2:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            with col3:
                top_k = st.slider("Top-k", 1, 100, 50)
            
            if st.button("🎯 Generate"):
                with st.spinner("Generating..."):
                    try:
                        output = generate_text(
                            model=st.session_state['model'],
                            tokenizer=st.session_state['tokenizer'],
                            prompt=prompt,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k
                        )
                        
                        st.subheader("Generated Text:")
                        st.text_area("Output:", value=output, height=200, disabled=True)
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
        else:
            st.info("Load a model to enable inference.")
    
    with tab3:
        st.header("Model & Checkpoint Management")
        
        # List available checkpoints
        checkpoint_base = "checkpoints"
        if os.path.exists(checkpoint_base):
            checkpoint_dirs = [d for d in os.listdir(checkpoint_base) 
                             if os.path.isdir(os.path.join(checkpoint_base, d))]
            
            if checkpoint_dirs:
                st.subheader("Available Models")
                
                for run_name in checkpoint_dirs:
                    checkpoint_path = os.path.join(checkpoint_base, run_name, "checkpoint.pt")
                    config_path = os.path.join(checkpoint_base, run_name, "config.json")
                    
                    if os.path.exists(checkpoint_path) and os.path.exists(config_path):
                        with st.expander(f"📁 {run_name}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Load and display config info
                                try:
                                    with open(config_path, 'r') as f:
                                        config_data = json.load(f)
                                    
                                    st.write(f"**Architecture:** {config_data.get('architecture_mode', 'N/A')}")
                                    st.write(f"**Training Mode:** {config_data.get('training_mode', 'N/A')}")
                                    st.write(f"**Experts:** {config_data.get('num_experts', 'N/A')}")
                                    if config_data.get('ghost', {}).get('num_ghost_experts', 0) > 0:
                                        st.write(f"**Ghost Experts:** {config_data['ghost']['num_ghost_experts']}")
                                    
                                except Exception as e:
                                    st.error(f"Error reading config: {e}")
                            
                            with col2:
                                if st.button(f"📥 Load {run_name}", key=f"load_{run_name}"):
                                    try:
                                        # Load the model
                                        with open(config_path, 'r') as f:
                                            config_dict = json.load(f)
                                        
                                        config = MoEConfig.from_dict(config_dict)
                                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                        
                                        # Initialize tokenizer
                                        tokenizer = AutoTokenizer.from_pretrained('gpt2')
                                        tokenizer.pad_token = tokenizer.eos_token
                                        
                                        # Create and load model
                                        model = MoEModel(config).to(device)
                                        optimizer = create_dynamic_optimizer(model, config)
                                        scheduler = PrimaryGhostLRScheduler(config, optimizer)
                                        
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
                                            'current_run_name': run_name,
                                            'model_info': {
                                                'architecture': config.architecture_mode,
                                                'parameters': f"{model.get_total_params()/1e6:.2f}M",
                                                'num_experts': config.num_experts,
                                                'num_ghost_experts': config.ghost.num_ghost_experts
                                            }
                                        })
                                        
                                        st.success(f"✅ Loaded model: {run_name}")
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Failed to load model: {str(e)}")
            else:
                st.info("No saved models found. Train a model to create checkpoints.")
        else:
            st.info("No checkpoints directory found.")
    
    with tab4:
        st.header("Dataset Management")
        
        # Dataset upload
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['json', 'jsonl', 'txt'],
            help="Upload a local dataset file (JSON, JSONL, or TXT format)"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            dataset_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Dataset saved to: {dataset_path}")
        
        # List existing datasets
        st.subheader("📁 Local Datasets")
        data_dir = "data"
        if os.path.exists(data_dir):
            dataset_files = [f for f in os.listdir(data_dir) 
                           if f.endswith(('.json', '.jsonl', '.txt'))]
            
            if dataset_files:
                for dataset_file in dataset_files:
                    with st.expander(f"📄 {dataset_file}"):
                        dataset_path = os.path.join(data_dir, dataset_file)
                        
                        # Show file info
                        file_size = os.path.getsize(dataset_path) / 1024  # KB
                        st.write(f"**Size:** {file_size:.1f} KB")
                        st.write(f"**Path:** {dataset_path}")
                        
                        # Preview button
                        if st.button(f"👁️ Preview", key=f"preview_{dataset_file}"):
                            try:
                                if dataset_file.endswith('.txt'):
                                    with open(dataset_path, 'r', encoding='utf-8') as f:
                                        lines = f.readlines()[:5]
                                    st.text("First 5 lines:")
                                    st.code('\n'.join(lines))
                                
                                else:  # JSON/JSONL
                                    import json
                                    with open(dataset_path, 'r', encoding='utf-8') as f:
                                        if dataset_file.endswith('.jsonl'):
                                            data = [json.loads(line) for line in f.readlines()[:3]]
                                        else:
                                            data = json.load(f)
                                            if isinstance(data, list):
                                                data = data[:3]
                                    
                                    st.json(data)
                                    
                            except Exception as e:
                                st.error(f"Error previewing file: {str(e)}")
            else:
                st.info("No dataset files found in the data directory.")
        else:
            st.info("Data directory not found.")
        
        # Preprocessed datasets
        st.subheader("⚡ Preprocessed Datasets")
        preprocessed_dir = "data-preprocessed"
        if os.path.exists(preprocessed_dir):
            from core.dataset_manager import DatasetManager
            
            dataset_manager = DatasetManager(preprocessed_dir)
            datasets = dataset_manager.list_datasets()
            
            if datasets:
                for dataset in datasets:
                    with st.expander(f"⚡ {dataset['name']}"):
                        st.write(f"**Size:** {dataset['size_mb']:.1f} MB")
                        st.write(f"**Created:** {dataset.get('created', 'N/A')}")
                        st.write(f"**Tokenizer:** {dataset.get('tokenizer', 'N/A')}")
                        st.write(f"**Max Length:** {dataset.get('max_length', 'N/A')}")
                        
                        if st.button(f"🗑️ Delete", key=f"delete_{dataset['name']}"):
                            if dataset_manager.delete_dataset(dataset['name']):
                                st.success("Dataset deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete dataset.")
            else:
                st.info("No preprocessed datasets found.")
        else:
            st.info("No preprocessed datasets directory found.")
    
    # Auto-refresh for live updates and process background training updates
    if get_training_state()['is_active']:
        # Process any pending training updates from background thread
        from utils.background_training import get_training_manager
        
        manager = get_training_manager()
        updates = manager.get_progress_updates()
        
        # Process different types of updates
        status_messages = []
        completion_message = None
        
        for update in updates:
            if update['type'] == 'metrics':
                from utils.state_management import add_training_datapoint
                add_training_datapoint(
                    step=update['step'],
                    epoch=update['epoch'],
                    loss=update['loss'],
                    geometric_metrics=update.get('geometric_metrics'),
                    ghost_metrics=update.get('ghost_metrics'),
                    expert_activations=update.get('expert_activations')
                )
            elif update['type'] == 'status':
                status_messages.append(update['message'])
                # Check for completion/stop messages
                if update.get('stage') in ['completed', 'stopped']:
                    completion_message = update['message']
            elif update['type'] == 'completion':
                # Training completed - show completion notification
                completion_message = update['message']
                if update.get('final_loss'):
                    completion_message += f" (Final Loss: {update['final_loss']:.4f})"
            elif update['type'] == 'error':
                st.error(update['message'])
        
        # Show recent status messages
        if status_messages:
            # Display in the sidebar below training controls
            with st.sidebar:
                with st.expander("📡 Training Status", expanded=True):
                    for message in status_messages[-3:]:  # Show last 3 status messages
                        st.write(message)
        
        # Show completion notification prominently
        if completion_message:
            if "completed successfully" in completion_message.lower():
                st.success(completion_message)
                st.balloons()  # Celebratory effect for completion
            elif "stopped" in completion_message.lower():
                st.warning(completion_message)
            else:
                st.info(completion_message)
        
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
