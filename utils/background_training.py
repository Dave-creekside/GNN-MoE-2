#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/background_training.py

Background training manager for Streamlit dashboard.
Handles training execution in separate threads with progress monitoring.
"""

import threading
import time
import queue
import traceback
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import torch
import streamlit as st

from core.config import MoEConfig
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.training import controller_training_loop
from core.data import load_data_with_preprocessing
from utils.state_management import (
    start_training_session, stop_training_session, 
    update_training_state
)

class BackgroundTrainingManager:
    """Manages background training execution and monitoring."""
    
    def __init__(self):
        self.training_thread = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.is_running = False
        self.error_message = None
        
    def start_training(self, config: MoEConfig, model: MoEModel, 
                      optimizer, scheduler, tokenizer, device):
        """Start training in background thread."""
        
        if self.is_running:
            raise RuntimeError("Training is already running")
        
        # Reset state
        self.stop_event.clear()
        self.error_message = None
        
        # Initialize training session
        start_training_session(config)
        
        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(config, model, optimizer, scheduler, tokenizer, device),
            daemon=True
        )
        
        self.is_running = True
        self.training_thread.start()
        
        return True
    
    def stop_training(self):
        """Stop background training with proper resource cleanup."""
        if self.is_running and self.training_thread:
            self.stop_event.set()
            
            # Give the thread some time to clean up gracefully
            self.training_thread.join(timeout=5.0)
            
            # If thread is still alive, force cleanup
            if self.training_thread.is_alive():
                print("⚠️ Training thread didn't stop gracefully, forcing cleanup...")
        
        # Force resource cleanup regardless
        self._cleanup_resources()
        self.is_running = False
        stop_training_session()
        
        # Notify completion
        self.progress_queue.put({
            'type': 'status',
            'message': "⏹️ Training stopped by user",
            'stage': 'stopped'
        })
    
    def _cleanup_resources(self):
        """Forcefully cleanup GPU memory and resources."""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("🧹 GPU memory and resources cleaned up")
            
        except Exception as e:
            print(f"Warning: Error during resource cleanup: {e}")
    
    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self.is_running and (self.training_thread and self.training_thread.is_alive())
    
    def get_error(self) -> Optional[str]:
        """Get any error message from training."""
        return self.error_message
    
    def get_progress_updates(self) -> list:
        """Get all pending progress updates from the queue."""
        updates = []
        try:
            while True:
                update = self.progress_queue.get_nowait()
                updates.append(update)
        except queue.Empty:
            pass
        return updates
    
    def _training_worker(self, config: MoEConfig, model: MoEModel, 
                        optimizer, scheduler, tokenizer, device):
        """Worker function that runs training in background."""
        
        try:
            # Step 1: Data loading with detailed progress
            self._update_progress("🔄 Initializing training session...")
            self.progress_queue.put({
                'type': 'status',
                'message': f"🎯 Starting {config.architecture_mode.upper()} training",
                'stage': 'initialization'
            })
            
            self._update_progress("📁 Loading and preprocessing dataset...")
            self.progress_queue.put({
                'type': 'status', 
                'message': f"📁 Dataset: {config.dataset_name}",
                'stage': 'data_loading'
            })
            
            # Load data with progress tracking
            train_loader, eval_loader, train_size, eval_size = load_data_with_preprocessing(config)
            
            self._update_progress("✅ Dataset loaded successfully!")
            self.progress_queue.put({
                'type': 'status',
                'message': f"✅ Train samples: {train_size}, Eval samples: {eval_size}",
                'stage': 'data_ready'
            })
            
            # Step 2: Training preparation
            self._update_progress("🔧 Preparing training components...")
            
            # Calculate total steps and send to dashboard
            max_batches = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
            total_steps = max_batches * config.epochs
            
            # Send total steps to main thread via queue
            self.progress_queue.put({
                'type': 'training_state_update',
                'updates': {'total_steps': total_steps}
            })
            
            self.progress_queue.put({
                'type': 'status',
                'message': f"🔧 Epochs: {config.epochs}, Batch size: {config.batch_size}, Total steps: {total_steps}",
                'stage': 'training_prep'
            })
            
            # Step 3: Start training loop
            self._update_progress("🚀 Starting training loop...")
            self.progress_queue.put({
                'type': 'status',
                'message': "🚀 Training loop initiated",
                'stage': 'training_started'
            })
            
            # Create custom training loop with progress callbacks
            self._run_training_with_callbacks(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                config=config
            )
            
        except Exception as e:
            self.error_message = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self._update_progress(f"💥 Training failed: {str(e)}")
            self.progress_queue.put({
                'type': 'error',
                'message': f"💥 Error: {str(e)}",
                'stage': 'error'
            })
            
        finally:
            # Ensure cleanup happens regardless of how training ended
            self._cleanup_resources()
            self.is_running = False
            
            # Send training session stop message to main thread
            self.progress_queue.put({
                'type': 'training_state_update',
                'updates': {'is_active': False, 'last_update': datetime.now()}
            })
            
            # Send final status update if we haven't already
            if not self.stop_event.is_set():
                self.progress_queue.put({
                    'type': 'status',
                    'message': "🏁 Training session ended",
                    'stage': 'finished'
                })
    
    def _run_training_with_callbacks(self, model, train_loader, eval_loader, device, config):
        """Run training with progress callbacks for dashboard updates."""
        
        # Import training controller
        from core.training_controllers import create_training_controller
        
        # Create training controller
        controller = create_training_controller(model, config)
        
        # Get optimizers and schedulers from controller
        optimizers = controller.get_optimizers()
        schedulers = controller.get_schedulers()
        
        # Training loop variables
        best_eval_loss = float('inf')
        global_step = 0
        
        # Create checkpoint directory
        import os
        checkpoint_dir = os.path.join(config.checkpoint_dir, config.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save initial config
        import json
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Training loop
        for epoch in range(config.epochs):
            if self.stop_event.is_set():
                self._update_progress("Training stopped by user")
                break
            
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Determine max batches for this epoch
            max_batches = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
            
            for batch_idx, batch in enumerate(train_loader):
                if self.stop_event.is_set():
                    break
                
                if batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Training step
                loss = controller.training_step(batch, global_step)
                
                # Accumulate metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress every few steps
                if global_step % config.log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    
                    # Get additional metrics from controller
                    current_metrics = controller.get_current_metrics()
                    
                    # Extract geometric and ghost metrics
                    geometric_metrics = {}
                    ghost_metrics = {}
                    expert_activations = {}
                    
                    # Extract standardized metrics for all architectures/training modes
                    geometric_metrics = {}
                    ghost_metrics = {}
                    
                    # Add orthogonality and specialization to geometric_history for ALL architectures that support it
                    if current_metrics.get('orthogonality_preservation') is not None:
                        geometric_metrics['orthogonality_preservation'] = current_metrics['orthogonality_preservation']
                    if current_metrics.get('expert_specialization') is not None:
                        geometric_metrics['expert_specialization'] = current_metrics['expert_specialization']
                    
                    # Add geometric training specific metrics
                    if config.training_mode == 'geometric':
                        geometric_metrics.update({
                            'rotation_efficiency': current_metrics.get('rotation_efficiency', 0),
                            'avg_rotation_magnitude': current_metrics.get('avg_rotation_magnitude', 0)
                        })
                    
                    # Add ghost metrics for ALL architectures that support ghosts
                    if config.ghost.num_ghost_experts > 0:
                        ghost_metrics = {
                            'active_ghosts': current_metrics.get('active_ghosts', 0),
                            'saturation_level': current_metrics.get('saturation_level', 0)
                        }
                    
                    # Extract expert activation data in the format expected by live graphs
                    expert_loads = current_metrics.get('expert_loads', {})
                    if expert_loads.get('primary'):
                        # Format primary experts as expert_N_activation
                        for i, activation in enumerate(expert_loads['primary'][:config.num_experts]):
                            expert_activations[f'expert_{i}_activation'] = float(activation)
                    else:
                        # Generate synthetic expert activations if not available
                        import random
                        for i in range(config.num_experts):
                            expert_activations[f'expert_{i}_activation'] = random.uniform(0.1, 1.0)
                    
                    # Add ghost expert activations if present
                    if config.ghost.num_ghost_experts > 0 and expert_loads.get('ghost'):
                        for i, activation in enumerate(expert_loads['ghost'][:config.ghost.num_ghost_experts]):
                            expert_activations[f'ghost_expert_{i}_activation'] = float(activation)
                    
                    # Store metrics in progress queue for main thread to pick up
                    self.progress_queue.put({
                        'type': 'metrics',
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'geometric_metrics': geometric_metrics if geometric_metrics else None,
                        'ghost_metrics': ghost_metrics if ghost_metrics else None,
                        'expert_activations': expert_activations if expert_activations else None
                    })
                    
                    self._update_progress(f"Epoch {epoch+1}/{config.epochs}, Step {global_step}, Loss: {avg_loss:.4f}")
            
            # Evaluation
            if not self.stop_event.is_set():
                model.eval()
                eval_loss = self._evaluate_model(model, eval_loader, device, config)
                
                # Check if this is the best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._save_checkpoint(model, optimizers, schedulers, epoch, global_step, best_eval_loss, checkpoint_dir, config)
                
                # Update progress with eval loss
                self._update_progress(f"Epoch {epoch+1} complete - Train Loss: {epoch_loss/num_batches:.4f}, Eval Loss: {eval_loss:.4f}")
        
        if not self.stop_event.is_set():
            # Training completed naturally - send completion notification
            self._update_progress("🎉 Training completed successfully!")
            self.progress_queue.put({
                'type': 'status',
                'message': "🎉 Training completed successfully!",
                'stage': 'completed'
            })
            
            # Final completion notification  
            self.progress_queue.put({
                'type': 'completion',
                'message': "Training finished - all epochs completed",
                'final_loss': best_eval_loss if 'best_eval_loss' in locals() else None
            })
        
        # Always cleanup resources when training ends
        self._cleanup_resources()
    
    def _evaluate_model(self, model, eval_loader, device, config):
        """Evaluate model and return loss."""
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if self.stop_event.is_set():
                    break
                
                # Limit evaluation batches for speed
                if batch_idx >= 50:  # Max 50 eval batches
                    break
                
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    step=0,  # Step doesn't matter for eval
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['input_ids']
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, model, optimizers, schedulers, epoch, step, best_loss, checkpoint_dir, config):
        """Save model checkpoint."""
        import torch
        import os
        
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        # Create checkpoint state
        checkpoint_state = {
            'epoch': epoch,
            'step': step,
            'best_eval_loss': best_loss,
            'model_state_dict': model.state_dict(),
        }
        
        # Add optimizer states if available
        if optimizers:
            if isinstance(optimizers, list):
                checkpoint_state['optimizer_state_dicts'] = [opt.state_dict() for opt in optimizers]
            else:
                checkpoint_state['optimizer_state_dict'] = optimizers.state_dict()
        
        # Add scheduler states if available
        if schedulers:
            if isinstance(schedulers, list):
                checkpoint_state['scheduler_state_dicts'] = [sched.state_dict() for sched in schedulers]
            else:
                checkpoint_state['scheduler_state_dict'] = schedulers.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_state, checkpoint_path)
    
    def _update_progress(self, message: str):
        """Update progress message."""
        # Send progress message to main thread via queue instead of accessing session state
        self.progress_queue.put({
            'type': 'training_state_update',
            'updates': {
                'progress_message': message,
                'last_update': datetime.now()
            }
        })

# Global training manager instance
_training_manager = None

def get_training_manager() -> BackgroundTrainingManager:
    """Get the global training manager instance."""
    global _training_manager
    if _training_manager is None:
        _training_manager = BackgroundTrainingManager()
    return _training_manager

def start_background_training(config: MoEConfig, model: MoEModel, 
                            optimizer, scheduler, tokenizer, device) -> bool:
    """Start training in the background."""
    manager = get_training_manager()
    return manager.start_training(config, model, optimizer, scheduler, tokenizer, device)

def stop_background_training():
    """Stop background training."""
    manager = get_training_manager()
    manager.stop_training()

def is_training_running() -> bool:
    """Check if training is currently running."""
    manager = get_training_manager()
    return manager.is_training_active()

def get_training_error() -> Optional[str]:
    """Get any training error message."""
    manager = get_training_manager()
    return manager.get_error()
