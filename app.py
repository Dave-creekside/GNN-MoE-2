#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py

An interactive CLI application for managing and experimenting with MoE models.
"""

import torch
import os
import json
from transformers import AutoTokenizer

from core.config import MoEConfig, HGNNParams, GhostParams
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.inference import generate_text
from core.training import standard_training_loop, load_checkpoint
from core.analysis import run_analysis
from core.data import load_data

# --- Global State ---
# This will hold the currently loaded model and its config
state = {
    "config": None,
    "model": None,
    "optimizer": None,
    "scheduler": None,
    "tokenizer": None,
    "device": None,
    "checkpoint_path": None,
    "start_epoch": 0,
    "resume_step": 0,
    "best_eval_loss": float('inf')
}

# --- Helper Functions ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    clear_screen()
    print("=" * 60)
    print(f"🧠 MoE Research Hub: {title}")
    print("=" * 60)

def print_model_status():
    if state["model"]:
        config = state["config"]
        print(f"\n--- Model Status ---")
        print(f"Loaded Model: {config.run_name}")
        print(f"Architecture: {config.architecture_mode}")
        if config.num_parameters is not None:
            print(f"Parameters: {config.num_parameters/1e6:.2f}M")
        else:
            print("Parameters: (Not calculated)")
        print(f"Checkpoint: {state['checkpoint_path']}")
        print("-" * 20 + "\n")
    else:
        print("\n--- Model Status ---")
        print("No model loaded.")
        print("-" * 20 + "\n")

# --- Menu Functions ---

def model_menu():
    """Menu for interacting with a loaded model."""
    while True:
        print_header("Model Menu")
        print_model_status()
        print("1. Run Inference")
        print("2. Continue Training")
        print("3. View Full Configuration")
        print("4. Generate Analysis Plots")
        print("5. Return to Main Menu")
        
        choice = input("> ")
        if choice == '1':
            run_inference_menu()
        elif choice == '2':
            continue_training_menu()
        elif choice == '3':
            view_config_menu()
        elif choice == '4':
            generate_plots_menu()
        elif choice == '5':
            return
        else:
            print("Invalid choice, please try again.")
            input("Press Enter to continue...")

def run_inference_menu():
    """Menu for running inference with the loaded model."""
    print_header("Run Inference")
    print_model_status()
    
    try:
        prompt = input("Enter your prompt: ")
        max_length = int(input("Enter max new tokens [100]: ") or 100)
        temperature = float(input("Enter temperature (e.g., 0.8) [0.8]: ") or 0.8)
        top_k = int(input("Enter top-k sampling (e.g., 50) [50]: ") or 50)

        output = generate_text(
            model=state["model"],
            tokenizer=state["tokenizer"],
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
        
        print("\n" + "="*20 + " GENERATED TEXT " + "="*20)
        print(output)
        print("="*56 + "\n")

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
    
    input("Press Enter to return to the Model Menu...")


def view_config_menu():
    """Displays the full configuration of the loaded model in a readable format."""
    print_header(f"Full Configuration: {state['config'].run_name}")
    
    if not state["config"]:
        print("No model loaded.")
        input("\nPress Enter to return to the Model Menu...")
        return

    try:
        config = state["config"]

        def print_section(title, data):
            print(f"\n--- {title} ---")
            if not data:
                print("  (No parameters in this section)")
                return
            
            if hasattr(data, '__dict__'):
                data = vars(data)

            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    continue
                print(f"  {key.replace('_', ' ').title()}: {value}")

        core_params = {
            'run_name': config.run_name,
            'architecture_mode': config.architecture_mode,
            'seed': config.seed,
            'num_parameters': config.num_parameters
        }
        model_params = {
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'num_experts': config.num_experts,
            'max_seq_length': config.max_seq_length,
            'vocab_size': config.vocab_size,
            'dropout_rate': config.dropout_rate
        }
        training_params = {
            'training_loop': config.training_loop,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_steps': config.max_steps,
            'eval_every': config.eval_every
        }
        dataset_params = {
            'dataset_source': config.dataset_source,
            'dataset_name': config.dataset_name,
            'dataset_config_name': config.dataset_config_name,
            'num_train_samples': config.num_train_samples,
            'num_eval_samples': config.num_eval_samples
        }

        print_section("Core Parameters", core_params)
        print_section("Model Architecture", model_params)
        
        if config.use_hypergraph_coupling:
            print_section("HGNN Parameters", config.hgnn)
        
        if config.ghost.num_ghost_experts > 0:
            print_section("Ghost Expert Parameters", config.ghost)
            
        print_section("Training Parameters", training_params)
        print_section("Dataset Parameters", dataset_params)

    except Exception as e:
        print(f"\nAn error occurred while displaying the configuration: {e}")

    input("\nPress Enter to return to the Model Menu...")


def generate_plots_menu():
    """Generates analysis plots for the loaded model's run."""
    print_header("Generate Analysis Plots")

    if not state["checkpoint_path"]:
        print("No model loaded.")
        input("\nPress Enter to return to the Model Menu...")
        return

    try:
        checkpoint_dir = os.path.dirname(state["checkpoint_path"])
        log_path = os.path.join(checkpoint_dir, "training_log.json")
        
        if not os.path.exists(log_path):
            print(f"Error: training_log.json not found in {checkpoint_dir}")
        else:
            print(f"Found log file: {log_path}")
            print("Generating plots...")
            run_analysis(log_path)
            print(f"\n✅ Plots have been generated in: {checkpoint_dir}")

    except Exception as e:
        print(f"An error occurred while generating plots: {e}")

    input("\nPress Enter to return to the Model Menu...")


def launch_training(train_loader=None, eval_loader=None):
    """
    A central function to set up and run a training session.
    """
    print_header("Launch Training")
    
    try:
        config = state["config"]
        
        if train_loader is None or eval_loader is None:
            print("🚀 Setting up data loading...")
            train_loader, eval_loader, _, _ = load_data(config)
        
        # Calculate max_steps if not provided, now that we have the dataloader
        if config.max_steps is None:
            actual_batches_per_epoch = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
            config.max_steps = config.epochs * actual_batches_per_epoch
            # We need to re-initialize the scheduler with the new max_steps
            state['scheduler'] = PrimaryGhostLRScheduler(config, state['optimizer'])


        print("🚀 Starting training...")
        if config.training_loop == "standard":
            training_loop_func = standard_training_loop
        else:
            print(f"Error: Unknown training loop '{config.training_loop}'")
            return

        training_loop_func(
            model=state["model"],
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=state["device"],
            config=config,
            resume_from_epoch=state["start_epoch"],
            resume_step=state["resume_step"],
            initial_best_loss=state["best_eval_loss"]
        )
        
        print("\n✅ Training finished. Running final analysis...")
        log_path = os.path.join(config.checkpoint_dir, "training_log.json")
        run_analysis(log_path)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

    input("\nPress Enter to return to the main menu...")


def continue_training_menu():
    """Confirms and launches a continued training session."""
    print_header("Continue Training")
    print_model_status()
    
    print("The model will continue training with its current configuration.")
    
    try:
        print("\n1. Continue with original dataset")
        print("2. Fine-tune on a new dataset")
        data_choice = input("> ")
        if data_choice == '2':
            state['config'] = edit_dataset_menu(state['config'])
            # Reset scheduler and optimizer for fine-tuning
            print("Resetting optimizer and scheduler for fine-tuning...")
            state['optimizer'] = create_dynamic_optimizer(state['model'], state['config'])
            state['scheduler'] = PrimaryGhostLRScheduler(state['config'], state['optimizer'])
            state['start_epoch'] = 0
            state['resume_step'] = 0
            state['best_eval_loss'] = float('inf')


        new_epochs = int(input(f"\nEnter total epochs to run to (current is {state['config'].epochs}): ") or state['config'].epochs)
        state['config'].epochs = new_epochs
        
        confirm = input("Start training? (y/n): ")
        if confirm.lower() == 'y':
            launch_training()
        else:
            print("Training cancelled.")
            input("Press Enter to continue...")

    except ValueError:
        print("Invalid number for epochs.")
        input("Press Enter to continue...")


def load_model_menu():
    """Menu for loading a model from a checkpoint."""
    print_header("Load Model")
    
    try:
        path = input("Enter the full path to the checkpoint.pt file (e.g., checkpoints/my_run/checkpoint.pt): ")
        if not os.path.exists(path):
            print("\nFile not found!")
            input("Press Enter to continue...")
            return

        checkpoint_dir = os.path.dirname(path)
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = MoEConfig.from_dict(config_dict)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        model = MoEModel(config).to(device)
        optimizer = create_dynamic_optimizer(model, config)
        # We need to calculate max_steps before creating the scheduler if it's not in the config
        if config.max_steps is None:
             print("Loading data to determine training steps for scheduler...")
             # This is a bit inefficient, but necessary to properly resume
             temp_train_loader, _, _, _ = load_data(config)
             actual_batches = len(temp_train_loader) if config.max_batches_per_epoch == -1 else min(len(temp_train_loader), config.max_batches_per_epoch)
             config.max_steps = config.epochs * actual_batches

        scheduler = PrimaryGhostLRScheduler(config, optimizer)

        start_epoch, resume_step, best_eval_loss = load_checkpoint(path, model, optimizer, scheduler)
        
        num_params = model.get_total_params()
        config.num_parameters = num_params

        state.update({
            "config": config, "model": model, "optimizer": optimizer,
            "scheduler": scheduler, "tokenizer": tokenizer, "device": device,
            "checkpoint_path": path, "start_epoch": start_epoch,
            "resume_step": resume_step, "best_eval_loss": best_eval_loss
        })
        
        print("\n✅ Model loaded successfully!")
        input("Press Enter to continue to the Model Menu...")
        model_menu()

    except Exception as e:
        print(f"\nAn error occurred while loading the model: {e}")
        input("Press Enter to continue...")


def edit_dataset_menu(config: MoEConfig):
    """UI for selecting a dataset source and path."""
    print_header("Configure Dataset")
    print("Select Dataset Source:")
    print("1. Hugging Face Hub")
    print("2. Local File (.txt, .json, .jsonl)")
    source_choice = input("> ")

    if source_choice == '1':
        config.dataset_source = "huggingface"
        path = input(f"Enter HF dataset path (e.g., wikitext/wikitext-2-v1) [{config.dataset_name}]: ") or config.dataset_name
        parts = path.split('/')
        config.dataset_name = parts[0]
        config.dataset_config_name = parts[1] if len(parts) > 1 else ""
    elif source_choice == '2':
        config.dataset_source = "local_file"
        path = input(f"Enter path to local file [{config.dataset_name}]: ") or config.dataset_name
        config.dataset_name = path
        config.dataset_config_name = ""
    else:
        print("Invalid choice.")
    return config

def advanced_config_menu(config: MoEConfig):
    """A menu to edit all parameters of a config object."""
    print_header("Advanced Configuration")
    
    try:
        config_dict = config.to_dict()
        # Flatten the config for editing
        params_to_edit = {**config_dict, **config_dict.pop('hgnn'), **config_dict.pop('ghost')}
        
        for key, value in params_to_edit.items():
            if key in ['num_parameters', 'hgnn', 'ghost', 'dataset_config_name']:
                continue
            
            new_value_str = input(f"Enter value for '{key}' (current: {value}): ")
            if new_value_str:
                original_type = type(value)
                if original_type == bool:
                    params_to_edit[key] = new_value_str.lower() in ['true', 't', '1', 'yes', 'y']
                else:
                    params_to_edit[key] = original_type(new_value_str)

        return MoEConfig.from_dict(params_to_edit)

    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to continue...")
        return config

def train_new_model_menu():
    """A wizard for configuring and launching a new training run."""
    config = MoEConfig()

    while True:
        clear_screen()
        print_header("Train New Model > Configuration Wizard")
        print("\nCurrent Configuration:")
        print(f" 1. Architecture: {config.architecture_mode}")
        print(f" 2. Run Name: {config.run_name}")
        print(f" 3. Batch Size: {config.batch_size}")
        print(f" 4. Num Experts: {config.num_experts}")
        print(f" 5. Num Ghost Experts: {config.ghost.num_ghost_experts if config.ghost else 0}")
        print(f" 6. Dataset: {config.dataset_source} -> {config.dataset_name}")
        print(f" 7. Advanced Configuration...")
        print("\n[S] Start Training with these settings")
        print("[E] Exit to Main Menu")
        
        choice = input("\nEnter a number to edit, or a command (S/E): ").lower()

        try:
            if choice == '1':
                mode = input(f"Enter architecture [gnn, hgnn, orthogonal, ghost] [{config.architecture_mode}]: ")
                if mode in ['gnn', 'hgnn', 'orthogonal', 'ghost']: config.architecture_mode = mode
            elif choice == '2':
                config.run_name = input(f"Enter run name [{config.run_name}]: ") or config.run_name
            elif choice == '3':
                config.batch_size = int(input(f"Enter batch size [{config.batch_size}]: ") or config.batch_size)
            elif choice == '4':
                config.num_experts = int(input(f"Enter num experts [{config.num_experts}]: ") or config.num_experts)
            elif choice == '5':
                config.ghost.num_ghost_experts = int(input(f"Enter num ghost experts [{config.ghost.num_ghost_experts}]: ") or config.ghost.num_ghost_experts)
            elif choice == '6':
                config = edit_dataset_menu(config)
            elif choice == '7':
                config = advanced_config_menu(config)
            elif choice == 's':
                config.__post_init__()
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                
                print("Instantiating model...")
                model = MoEModel(config).to(device)
                config.num_parameters = model.get_total_params()
                print(f"✅ Model initialized with {config.num_parameters/1e6:.2f}M parameters.")

                optimizer = create_dynamic_optimizer(model, config)
                
                state.update({
                    "config": config, "model": model, "optimizer": optimizer,
                    "scheduler": None, "tokenizer": tokenizer, "device": device,
                    "checkpoint_path": os.path.join(config.checkpoint_dir, config.run_name, "checkpoint.pt"),
                    "start_epoch": 0, "resume_step": 0, "best_eval_loss": float('inf')
                })
                
                launch_training()
                return
            elif choice == 'e':
                return
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            input("Press Enter to continue...")

def main_menu():
    """The main menu of the application."""
    while True:
        print_header("Main Menu")
        print_model_status()
        print("1. Train New Model")
        print("2. Load Model from Checkpoint")
        print("3. Exit")
        
        choice = input("> ")
        
        if choice == '1':
            train_new_model_menu()
        elif choice == '2':
            load_model_menu()
        elif choice == '3':
            break
        else:
            print("Invalid choice, please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main_menu()
