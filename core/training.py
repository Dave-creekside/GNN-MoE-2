#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py

Unified training loop, evaluation, and checkpointing for all MoE models.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import shutil
import json
from tqdm import tqdm
from collections import defaultdict
import time
import numpy as np

from .config import MoEConfig
from .architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from .training_controllers import create_training_controller
from .graceful_exit import setup_graceful_exit, check_exit_requested, cleanup_and_exit
from .data import load_data_with_preprocessing

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pt"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    # Save config.json every time (not just when is_best)
    if 'config' in state:
        config_dict = state['config']
        config_json_path = os.path.join(checkpoint_dir, "config.json")
        try:
            with open(config_json_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            print(f"Error saving config.json: {e}")
    
    # Save best model separately
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        shutil.copyfile(filepath, best_model_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")
        return 0, 0, float('inf')
    
    # The new torch version requires this for loading pickled custom classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    step = checkpoint.get('step', 0)
    
    return start_epoch, step, best_eval_loss

def prepare_batch(batch, device):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0 
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, config, max_batches=-1):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_eval_batches = len(eval_loader) if max_batches == -1 else min(len(eval_loader), max_batches)

    if num_eval_batches == 0:
        return float('inf'), float('inf')

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_eval_batches:
                break
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, step=0, attention_mask=attention_mask, labels=labels)
            
            mask = (labels != 0)
            if mask.sum().item() > 0:
                total_loss += outputs['loss'].item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity

def update_parameter_group_lrs(optimizer, primary_lr, ghost_lrs):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'primary_experts':
            param_group['lr'] = primary_lr
        elif param_group['name'] == 'ghost_experts' and ghost_lrs:
            param_group['lr'] = ghost_lrs[0]

def ensure_json_serializable(value):
    """OPTIMIZED: Single global function for JSON serialization."""
    if torch.is_tensor(value): 
        return value.cpu().detach().numpy().tolist()
    if isinstance(value, np.ndarray): 
        return value.tolist()
    return value

class BatchedLogger:
    """OPTIMIZED: Batched logging to reduce I/O overhead."""
    
    def __init__(self, checkpoint_dir, batch_size=50):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.log_buffer = []
        self.log_path = os.path.join(checkpoint_dir, 'training_log.json')
        
    def add_entry(self, log_entry):
        """Add log entry to buffer."""
        self.log_buffer.append(log_entry)
        
        # Flush buffer if it's full
        if len(self.log_buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write buffered logs to disk."""
        if not self.log_buffer:
            return
            
        # Load existing logs if they exist
        existing_logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    existing_logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_logs = []
        
        # Append new logs
        existing_logs.extend(self.log_buffer)
        
        # Write back to file
        with open(self.log_path, 'w') as f:
            json.dump(existing_logs, f, indent=4)
        
        # Clear buffer
        self.log_buffer.clear()
    
    def get_all_logs(self):
        """Get all logs (buffered + persisted)."""
        all_logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    all_logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_logs = []
        
        return all_logs + self.log_buffer

def log_training_metrics(step, total_loss, base_loss, orthogonality_loss, model, primary_lr, ghost_lrs):
    print(f"Step {step}: Total Loss: {total_loss.item():.4f}, Base Loss: {base_loss.item():.4f}, Ortho Loss: {orthogonality_loss.item():.4f}")
    if model.config.ghost.num_ghost_experts > 0:
        ghost_activations = model.get_current_ghost_activations()
        saturation_metrics = model.get_last_saturation_metrics()
        print(f"  LRs - Primary: {primary_lr:.2e}, Ghosts: {[f'{lr:.2e}' for lr in ghost_lrs]}")
        print(f"  Ghost Activations: {[f'{act:.2f}' for act in ghost_activations]}")
        if saturation_metrics:
            print(f"  Saturation: {saturation_metrics.get('saturation_level', 0):.4f}, Ortho Score: {saturation_metrics.get('orthogonality_score', 0):.4f}")

def standard_training_loop(model: MoEModel, optimizer, scheduler, train_loader, eval_loader, device, config: MoEConfig,
                           resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    
    actual_batches_per_epoch = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
    total_steps = config.max_steps
    
    if total_steps == 0:
        print("Warning: total_steps is 0. No training will occur.")
        return [], float('inf')

    # Setup graceful exit monitoring
    exit_monitor = setup_graceful_exit()

    training_log = []
    best_eval_loss = initial_best_loss
    current_step = resume_step
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(resume_from_epoch, config.epochs):
        model.train()
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch)

        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx >= actual_batches_per_epoch:
                break

            # Check for graceful exit request
            if check_exit_requested():
                cleanup_and_exit(model, optimizer, scheduler, config, current_step, epoch, best_eval_loss)
                return training_log, best_eval_loss

            input_ids, attention_mask, labels = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, step=current_step, attention_mask=attention_mask, return_loss=True, labels=labels)
            
            base_loss = outputs['loss']
            orthogonality_loss = model.get_total_orthogonality_loss(current_step)
            total_loss = base_loss + orthogonality_loss
            
            total_loss.backward()
            
            optimizer.step()

            ghost_activations = model.get_current_ghost_activations()
            primary_lr, ghost_lrs = scheduler.step(ghost_activations)
            
            update_parameter_group_lrs(optimizer, primary_lr, ghost_lrs)
            current_step += 1
            
            pbar_train.set_postfix({'loss': f'{total_loss.item():.3f}', 'lr': f"{primary_lr:.1e}"})

            def ensure_json_serializable(value):
                if torch.is_tensor(value): return value.cpu().detach().numpy().tolist()
                if isinstance(value, np.ndarray): return value.tolist()
                return value

            # --- Frequent, Lightweight Logging ---
            if current_step % config.log_every == 0:
                expert_connections = {}
                if config.use_hypergraph_coupling and hasattr(model.model_layers[0], 'coupler'):
                    adj_matrix = model.model_layers[0].coupler.get_adjacency_matrix()
                    expert_connections['adjacency_matrix'] = ensure_json_serializable(adj_matrix)

                log_entry = {
                    'step': int(current_step),
                    'train_loss': float(total_loss.item()),
                    'primary_lr': float(primary_lr),
                    'ghost_lrs': ensure_json_serializable(ghost_lrs),
                    'ghost_activations': ensure_json_serializable(model.get_current_ghost_activations()),
                    'saturation_level': model.get_last_saturation_metrics().get('saturation_level', 0.0),
                    'orthogonality_score': model.get_last_saturation_metrics().get('orthogonality_score', 0.0),
                    'expert_connections': expert_connections,
                    'expert_loads': model.get_expert_activation_loads()
                }
                training_log.append(log_entry)

            # --- Evaluation and Checkpointing (Less Frequent) ---
            if current_step % config.eval_every == 0:
                print()
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config)
                
                log_training_metrics(current_step, total_loss, base_loss, orthogonality_loss, model, primary_lr, ghost_lrs)
                print(f"  Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

                # Find the corresponding log entry to update with eval results
                for log in reversed(training_log):
                    if log['step'] == current_step:
                        log['eval_loss'] = float(eval_loss)
                        log['eval_perplexity'] = float(perplexity)
                        break
                
                log_path = os.path.join(config.checkpoint_dir, 'training_log.json')
                with open(log_path, 'w') as f:
                    json.dump(training_log, f, indent=4)
                
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                
                config_dict = config.to_dict()
                save_checkpoint({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_eval_loss': best_eval_loss,
                    'config': config_dict
                }, is_best, checkpoint_dir=config.checkpoint_dir)
                
                model.train()

    log_path = os.path.join(config.checkpoint_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"Saved final training log to {log_path}")

    return training_log, best_eval_loss


def train_model(model, optimizer, scheduler, train_loader, eval_loader, device, config, 
                resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    """
    Main training function that dispatches to the appropriate training loop based on config.
    """
    
    # Determine which training system to use
    if hasattr(config, 'training_mode') and config.training_mode == "geometric":
        # Use new controller-based training system
        return controller_training_loop(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            config=config,
            resume_from_epoch=resume_from_epoch,
            resume_step=resume_step,
            initial_best_loss=initial_best_loss
        )
    else:
        # Use legacy standard training system
        return standard_training_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            config=config,
            resume_from_epoch=resume_from_epoch,
            resume_step=resume_step,
            initial_best_loss=initial_best_loss
        )


def controller_training_loop(model: MoEModel, train_loader, eval_loader, device, config: MoEConfig,
                           resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    """New training loop using the training controller pattern."""
    
    # Create training controller based on config
    training_controller = create_training_controller(model, config)
    
    actual_batches_per_epoch = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
    total_steps = config.max_steps
    
    if total_steps == 0:
        print("Warning: total_steps is 0. No training will occur.")
        return [], float('inf')

    # Setup graceful exit monitoring
    exit_monitor = setup_graceful_exit()

    training_log = []
    best_eval_loss = initial_best_loss
    current_step = resume_step
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print(f"🚀 Using {config.training_mode} training controller")

    for epoch in range(resume_from_epoch, config.epochs):
        model.train()
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch)

        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx >= actual_batches_per_epoch:
                break

            # Check for graceful exit request
            if check_exit_requested():
                # For controller training, get optimizers and schedulers from controller
                controller_optimizers = training_controller.get_optimizers()
                controller_schedulers = training_controller.get_schedulers()
                optimizer = controller_optimizers[0] if controller_optimizers else None
                scheduler = controller_schedulers[0] if controller_schedulers else None
                cleanup_and_exit(model, optimizer, scheduler, config, current_step, epoch, best_eval_loss)
                return training_log, best_eval_loss

            # Prepare batch data
            batch_dict = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            
            # Controller-specific training step
            loss = training_controller.training_step(batch_dict, current_step)
            current_step += 1
            
            # Get current metrics from controller
            controller_metrics = training_controller.get_current_metrics()
            current_lr = controller_metrics.get('learning_rate', config.learning_rate)
            
            pbar_train.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f"{current_lr:.1e}"})

            # --- Frequent, Lightweight Logging ---
            if current_step % config.log_every == 0:
                log_entry = {
                    'step': int(current_step),
                    'train_loss': float(loss.item()),
                    'training_mode': config.training_mode,
                    'controller_metrics': {k: ensure_json_serializable(v) for k, v in controller_metrics.items()}
                }
                if hasattr(model, 'get_expert_activation_loads'):
                    log_entry['expert_loads'] = model.get_expert_activation_loads()
                if hasattr(model, 'get_current_ghost_activations'):
                    log_entry['ghost_activations'] = ensure_json_serializable(model.get_current_ghost_activations())
                training_log.append(log_entry)

            # --- Evaluation and Checkpointing (Less Frequent) ---
            if current_step % config.eval_every == 0:
                print()
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config)
                
                # Enhanced logging with controller metrics
                print(f"Step {current_step}: Training Loss: {loss.item():.4f}")
                print(f"  Controller: {config.training_mode}")
                print(f"  Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                
                # Print controller-specific metrics
                for key, value in controller_metrics.items():
                    if key not in ['loss', 'learning_rate']:
                        print(f"  {key.replace('_', ' ').title()}: {value}")

                # Find the corresponding log entry to update with eval results
                for log in reversed(training_log):
                    if log['step'] == current_step:
                        log['eval_loss'] = float(eval_loss)
                        log['eval_perplexity'] = float(perplexity)
                        break
                
                log_path = os.path.join(config.checkpoint_dir, 'training_log.json')
                with open(log_path, 'w') as f:
                    json.dump(training_log, f, indent=4)
                
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                
                config_dict = config.to_dict()
                
                # Save checkpoint with controller state
                checkpoint_state = {
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'best_eval_loss': best_eval_loss,
                    'config': config_dict,
                    'training_mode': config.training_mode
                }
                
                # Add controller optimizers and schedulers to checkpoint
                controller_optimizers = training_controller.get_optimizers()
                controller_schedulers = training_controller.get_schedulers()
                
                if controller_optimizers:
                    checkpoint_state['controller_optimizer_states'] = [opt.state_dict() for opt in controller_optimizers]
                if controller_schedulers:
                    checkpoint_state['controller_scheduler_states'] = [
                        sched.state_dict() if hasattr(sched, 'state_dict') else None 
                        for sched in controller_schedulers
                    ]
                
                save_checkpoint(checkpoint_state, is_best, checkpoint_dir=config.checkpoint_dir)
                
                model.train()

    log_path = os.path.join(config.checkpoint_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"Saved final training log to {log_path}")

    return training_log, best_eval_loss
