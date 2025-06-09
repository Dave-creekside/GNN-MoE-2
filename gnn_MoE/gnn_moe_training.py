#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_training.py

Training loop, evaluation, and checkpointing utilities for GNN-Coupled MoE models.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F # For F.cross_entropy if used directly, though model handles loss
from torch.utils.data import DataLoader # For type hinting
import math
import os
import shutil
from tqdm import tqdm
from collections import defaultdict
import time

# Assuming GNNMoEConfig and model classes will be imported in the main script
# from gnn_moe_config import GNNMoEConfig
# from gnn_moe_architecture import GNNMoEModel

# --- Checkpoint Helper Functions ---
def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    """Saves model and training parameters."""
    # Ensure checkpoint_dir exists (it should be created by main script based on args or defaults)
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory {checkpoint_dir} does not exist. Attempting to create.")
        try:
            os.makedirs(checkpoint_dir)
            print(f"Successfully created checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            print(f"Error creating checkpoint directory {checkpoint_dir}: {e}. Checkpoint will not be saved.")
            return

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pth.tar"))
        print(f"✅ Saved new best model to {os.path.join(checkpoint_dir, 'best_model.pth.tar')}")
    # else: # Reduce verbosity for non-best checkpoints
        # print(f"✅ Saved checkpoint to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Loads model and training parameters."""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint file not found at '{checkpoint_path}'. Starting from scratch.")
        return 0, 0, float('inf') # epoch, step, best_loss
    
    print(f"🔄 Loading checkpoint from '{checkpoint_path}'")
    try:
        # weights_only=False is needed if 'config': GNNMoEConfig_instance is saved in checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) 
    except RuntimeError as e:
        if "GLOBAL __main__.GNNMoEConfig" in str(e) or "GLOBAL gnn_moe_config.GNNMoEConfig" in str(e):
            print(f"Info: GNNMoEConfig class not found by torch.load. This can happen if the class definition changed or is not in scope.")
            print("Attempting to load with weights_only=True as a fallback for model state_dict only.")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                # If this succeeds, we can only restore model weights, not optimizer/scheduler/epoch etc.
                model.load_state_dict(checkpoint) # Assuming the checkpoint IS the state_dict
                print("✅ Loaded model weights only (weights_only=True fallback). Optimizer, scheduler, epoch not restored.")
                return 0, 0, float('inf') # Cannot reliably resume, so start fresh conceptually
            except Exception as e_fallback:
                print(f"❌ Fallback load with weights_only=True also failed: {e_fallback}")
                raise e # Re-raise original error
        else:
            raise e # Re-raise other runtime errors

    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    step = checkpoint.get('step', 0)
    # loaded_config = checkpoint.get('config') # Could compare if needed
    
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}, step {step}, best_eval_loss {best_eval_loss:.4f}")
    return start_epoch, step, best_eval_loss

# --- Training Utilities ---
def prepare_batch(batch, device):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0 
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, config, max_batches=-1): # max_batches=-1 for full eval
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Determine number of batches to evaluate
    if max_batches == -1: # Evaluate on the full validation set
        num_eval_batches = len(eval_loader)
    else: # Evaluate on a subset for speed (e.g., during training)
        num_eval_batches = min(len(eval_loader), max_batches)

    if num_eval_batches == 0:
        print("⚠️ Eval loader is empty. Skipping evaluation.")
        return float('inf'), float('inf')

    pbar_eval = tqdm(eval_loader, desc="Evaluating", leave=False, total=num_eval_batches)
    with torch.no_grad():
        for i, batch in enumerate(pbar_eval):
            if i >= num_eval_batches: # Ensure we don't exceed specified/available batches
                break
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, attention_mask, labels=labels) # Model computes loss
            
            # Calculate loss only on non-padding tokens for perplexity
            mask = (labels != 0)
            if mask.sum().item() > 0:
                total_loss += outputs['loss'].item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20)) # Cap perplexity for display stability
    return avg_loss, perplexity

def train_gnn_moe(model, train_loader, eval_loader, device, config,
                  resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Determine actual batches per epoch
    if config.max_batches_per_epoch == -1: # Use full epoch
        actual_batches_per_epoch = len(train_loader)
    else:
        actual_batches_per_epoch = min(len(train_loader), config.max_batches_per_epoch)
    
    total_steps = config.epochs * actual_batches_per_epoch
    
    if total_steps == 0:
        print("⚠️ Total training steps is 0. Ensure train_loader is not empty and epochs/max_batches_per_epoch are > 0.")
        return defaultdict(list), float('inf') # Return empty stats and inf loss
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6) # eta_min for smoother end
    
    # If resuming, load_checkpoint should have updated optimizer and scheduler states if they were saved.
    # If resume_step > 0 and scheduler state wasn't loaded, it might need manual stepping.
    # For simplicity, assume load_checkpoint handles this if states are in checkpoint.
    # If only model weights are loaded, optimizer/scheduler start fresh but step count is correct.
    for _ in range(resume_step): # Advance scheduler if resuming and scheduler state wasn't loaded
        if scheduler and (not hasattr(scheduler, '_step_count') or scheduler._step_count < resume_step +1 ): # crude check
             scheduler.step()


    stats = defaultdict(list)
    best_eval_loss = initial_best_loss
    current_step = resume_step 
    start_time = time.time()

    print(f"\n🚀 Starting/Resuming GNN-MoE Training on {device}")
    if resume_from_epoch > 0 or resume_step > 0:
        print(f"🔄 Resuming from epoch {resume_from_epoch}, global step {resume_step}. Initial best_eval_loss: {initial_best_loss:.4f}")
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🎯 Training: {config.epochs} epochs × {actual_batches_per_epoch} batches/epoch = {total_steps} total steps")
    print(f"💾 Checkpoints will be saved in: {config.checkpoint_dir}")

    for epoch in range(resume_from_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Determine start batch for this epoch if resuming
        # If current_step is 0, start_batch_idx is 0.
        # If current_step > 0 and it's the first epoch of resumption, skip already processed batches.
        start_batch_idx = (current_step % actual_batches_per_epoch) if (epoch == resume_from_epoch and current_step > 0 and current_step < total_steps) else 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch, initial=start_batch_idx)

        if start_batch_idx > 0:
            print(f"Epoch {epoch+1}: Resuming, skipping to batch index {start_batch_idx} (global step {current_step}).")
        
        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx < start_batch_idx:
                continue # Skip batches already processed in a resumed epoch
            
            if batch_idx >= actual_batches_per_epoch: # Should not happen if pbar.total is correct
                break

            input_ids, attention_mask, labels = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step() # Step scheduler after optimizer

            current_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            
            stats['train_loss'].append(loss.item())
            stats['grad_norm'].append(grad_norm.item())
            stats['learning_rate'].append(scheduler.get_last_lr()[0] if scheduler else config.learning_rate)

            tokens_processed = current_step * config.batch_size * config.max_seq_length
            elapsed_train = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed_train if elapsed_train > 0 else 0
            
            pbar_train.set_postfix({
                'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm.item():.2f}',
                'tok/s': f'{tokens_per_sec:.0f}', 'lr': f"{scheduler.get_last_lr()[0] if scheduler else config.learning_rate:.1e}"
            })

            if current_step % config.eval_every == 0 or current_step == total_steps:
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config, max_batches=-1) # Full eval
                stats['eval_loss'].append(eval_loss)
                stats['eval_perplexity'].append(perplexity)
                stats['eval_step'].append(current_step)
                print(f"\n📈 Step {current_step}: Eval Loss: {eval_loss:.4f}, PPL: {perplexity:.2f}")
                
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                    print(f"🎯 New best eval loss: {best_eval_loss:.4f}")
                
                save_checkpoint({
                    'epoch': epoch + 1, # Save as next epoch to start from
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_eval_loss': best_eval_loss,
                    'config': vars(config) # Save config as dict for portability
                }, is_best, checkpoint_dir=config.checkpoint_dir)
                model.train() # Set back to train mode
        
        # Reset start_batch_idx for subsequent epochs after a resumed one
        start_batch_idx = 0 

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('nan')
        print(f"Epoch {epoch+1} Summary - Avg Train Loss: {avg_epoch_loss:.4f}, Time: {(time.time() - start_time)/60:.1f}m")
        
        # Save checkpoint at end of each epoch as well (not necessarily the best)
        save_checkpoint({
            'epoch': epoch + 1, 
            'step': current_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_eval_loss': best_eval_loss,
            'config': vars(config)
        }, False, checkpoint_dir=config.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

    total_time_min = (time.time() - start_time) / 60
    print(f"\n✅ GNN-MoE Training Complete!")
    print(f"🎯 Best eval loss during this run: {best_eval_loss:.4f}")
    print(f"⏱️ Total time for this run: {total_time_min:.1f} minutes")
    return stats, best_eval_loss
