#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graceful_exit.py

Graceful exit system for training loops.
Allows users to press 'q' to exit cleanly without zombie processes.
"""

import threading
import sys
import select
import termios
import tty
import os
import torch
from typing import Optional


class GracefulExitMonitor:
    """
    Monitors for 'q' key press during training to enable graceful exit.
    Runs in background thread to avoid blocking training loop.
    """
    
    def __init__(self):
        self.should_exit = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.original_settings = None
        
    def start_monitoring(self):
        """Start the keyboard monitoring in a background thread."""
        if self.running:
            return
            
        self.should_exit = False
        self.running = True
        
        # Save original terminal settings (Unix/macOS)
        if hasattr(termios, 'tcgetattr'):
            try:
                self.original_settings = termios.tcgetattr(sys.stdin)
            except:
                self.original_settings = None
        
        self.monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self.monitor_thread.start()
        
        print("🔄 Training started. Press 'q' + Enter to exit gracefully...")
    
    def stop_monitoring(self):
        """Stop the keyboard monitoring and restore terminal settings."""
        self.running = False
        
        # Restore original terminal settings
        if self.original_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
            except:
                pass
    
    def _monitor_keyboard(self):
        """Monitor keyboard input in background thread."""
        try:
            while self.running:
                # Use a timeout to periodically check if we should stop
                if self._check_input_available():
                    try:
                        char = sys.stdin.read(1).lower()
                        if char == 'q':
                            print("\n🛑 Exit requested! Finishing current batch gracefully...")
                            self.should_exit = True
                            break
                    except:
                        # Handle any input errors gracefully
                        pass
                
                # Small sleep to prevent busy waiting
                threading.Event().wait(0.1)
                
        except Exception as e:
            # Silently handle any monitoring errors
            pass
    
    def _check_input_available(self) -> bool:
        """Check if input is available without blocking."""
        try:
            # Use select to check if stdin has data available
            if hasattr(select, 'select'):
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                return bool(ready)
            else:
                # Fallback for systems without select
                return False
        except:
            return False
    
    def cleanup_and_exit(self, model, optimizer, scheduler, config, current_step, epoch, best_loss):
        """
        Perform graceful cleanup and save emergency checkpoint.
        
        Args:
            model: The training model
            optimizer: Model optimizer
            scheduler: Learning rate scheduler  
            config: Training configuration
            current_step: Current training step
            epoch: Current epoch
            best_loss: Best validation loss so far
        """
        print("\n📝 Saving emergency checkpoint...")
        
        try:
            # Create emergency checkpoint directory
            emergency_dir = os.path.join(config.checkpoint_dir, "emergency_exit")
            os.makedirs(emergency_dir, exist_ok=True)
            
            # Save emergency checkpoint
            checkpoint_path = os.path.join(emergency_dir, f"emergency_step_{current_step}.pt")
            
            checkpoint_state = {
                'epoch': epoch,
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'best_eval_loss': best_loss,
                'config': config.to_dict(),
                'exit_reason': 'user_requested_graceful_exit'
            }
            
            # Add optimizer and scheduler if available
            if optimizer is not None:
                checkpoint_state['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None and hasattr(scheduler, 'state_dict'):
                checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_state, checkpoint_path)
            print(f"✅ Emergency checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save emergency checkpoint: {e}")
        
        # Clear GPU memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 CUDA cache cleared")
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print("🧹 MPS cache cleared")
        except Exception as e:
            print(f"⚠️  Warning: Could not clear GPU cache: {e}")
        
        # Stop monitoring
        self.stop_monitoring()
        
        print("\n✨ Graceful exit completed!")
        print(f"📊 Training Summary:")
        print(f"   Completed Steps: {current_step}")
        print(f"   Epoch: {epoch}")
        print(f"   Best Loss: {best_loss:.4f}")
        print(f"   Emergency checkpoint: {checkpoint_path}")
        
        return True


# Global instance for easy access
exit_monitor = GracefulExitMonitor()


def setup_graceful_exit():
    """Initialize and start the graceful exit monitor."""
    exit_monitor.start_monitoring()
    return exit_monitor


def check_exit_requested() -> bool:
    """Check if graceful exit has been requested."""
    return exit_monitor.should_exit


def cleanup_and_exit(model, optimizer=None, scheduler=None, config=None, current_step=0, epoch=0, best_loss=float('inf')):
    """Perform graceful cleanup and exit."""
    return exit_monitor.cleanup_and_exit(model, optimizer, scheduler, config, current_step, epoch, best_loss)
