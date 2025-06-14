#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py

An interactive CLI application for managing and experimenting with MoE models.
"""

import torch
import os
import json
import warnings
from transformers import AutoTokenizer

# --- Suppress Warnings ---
# Suppress NotOpenSSLWarning
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.config import MoEConfig, HGNNParams, GhostParams
from core.architecture import MoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from core.inference import generate_text
from core.training import standard_training_loop, controller_training_loop, load_checkpoint
from core.analysis import run_analysis
from core.data import load_data

# --- ANSI Color Codes ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def color_text(text, color):
    return f"{color}{text}{Colors.RESET}"

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
    while True:
        print_header("Run Inference")
        print_model_status()
        
        try:
            prompt = input("Enter your prompt (or 'b' to go back): ")
            if prompt.lower() == 'b':
                return

            max_length_str = input("Enter max new tokens [100]: ")
            max_length = int(max_length_str) if max_length_str else 100

            temperature_str = input("Enter temperature (e.g., 0.8) [0.8]: ")
            temperature = float(temperature_str) if temperature_str else 0.8

            top_k_str = input("Enter top-k sampling (e.g., 50) [50]: ")
            top_k = int(top_k_str) if top_k_str else 50

            print("\nGenerating text...")
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

        except ValueError:
            print("\nInvalid input. Please enter a valid number.")
        except Exception as e:
            print(f"\nAn error occurred during inference: {e}")
        
        input("Press Enter to continue...")


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
        
        # Select training system based on configuration
        if config.training_mode == "geometric" or config.training_loop == "controller":
            # Use new controller-based training system
            training_loop_func = controller_training_loop
            controller_training_loop(
                model=state["model"],
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=state["device"],
                config=config,
                resume_from_epoch=state["start_epoch"],
                resume_step=state["resume_step"],
                initial_best_loss=state["best_eval_loss"]
            )
        elif config.training_loop == "standard":
            # Use legacy training system
            standard_training_loop(
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
        else:
            print(f"Error: Unknown training loop '{config.training_loop}'")
            return
        
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
        path = input("Enter path to checkpoint.pt (or 'b' to go back): ")
        if path.lower() == 'b':
            return
            
        if not os.path.exists(path):
            print(f"\n❌ Error: File not found at '{path}'")
            input("Press Enter to continue...")
            return

        print("\nLoading model... this may take a moment.")
        
        checkpoint_dir = os.path.dirname(path)
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"\n❌ Error: config.json not found in '{checkpoint_dir}'")
            input("Press Enter to continue...")
            return
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = MoEConfig.from_dict(config_dict)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        model = MoEModel(config).to(device)
        optimizer = create_dynamic_optimizer(model, config)
        
        # Calculate max_steps for scheduler if needed
        if config.max_steps is None:
             print("Loading data to determine training steps for scheduler...")
             try:
                 temp_train_loader, _, _, _ = load_data(config)
                 actual_batches = len(temp_train_loader) if config.max_batches_per_epoch == -1 else min(len(temp_train_loader), config.max_batches_per_epoch)
                 config.max_steps = config.epochs * actual_batches
             except Exception as e:
                 print(f"\n❌ Error loading data to configure scheduler: {e}")
                 input("Press Enter to continue...")
                 return

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
        input("Press Enter to continue to the Main Menu...")

    except json.JSONDecodeError as e:
        print(f"\n❌ Error: Invalid JSON in config file: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"\nAn unexpected error occurred while loading the model: {e}")
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
        path = input(f"Enter HF dataset name (e.g., openai/gsm8k, Creekside/GRPO-Lambda-ParsedForUnsloth) [{config.dataset_name}]: ") or config.dataset_name
        # Strip quotes and whitespace from input
        path = path.strip().strip('"').strip("'").strip()
        
        # Treat the entire input as the dataset name by default
        config.dataset_name = path
        config.dataset_config_name = ""
        
        # Optional: Ask for config if user wants to specify one
        config_input = input("Enter config name (leave blank for default): ").strip()
        if config_input:
            config.dataset_config_name = config_input
    elif source_choice == '2':
        config.dataset_source = "local_file"
        path = input(f"Enter path to local file [{config.dataset_name}]: ") or config.dataset_name
        # Strip quotes and whitespace from input
        path = path.strip().strip('"').strip("'").strip()
        config.dataset_name = path
        config.dataset_config_name = ""
    else:
        print("Invalid choice.")
    return config

def advanced_config_menu(config: MoEConfig):
    """A menu to edit all parameters of a config object."""
    print_header("Advanced Configuration")
    
    try:
        # Training mode selection
        training_mode_choice = input(f"Training mode [standard/geometric] [{config.training_mode}]: ")
        if training_mode_choice in ['standard', 'geometric']:
            config.training_mode = training_mode_choice
        
        # If geometric mode selected, show geometric options
        if config.training_mode == "geometric":
            config = edit_geometric_config(config)
        
        # If ghost experts are configured, show ghost options
        if config.ghost.num_ghost_experts > 0:
            config = edit_ghost_config(config)
        
        # Continue with other advanced options
        config_dict = config.to_dict()
        # Flatten the config for editing, excluding nested configs
        params_to_edit = {**config_dict}
        params_to_edit.pop('hgnn', None)
        params_to_edit.pop('ghost', None)  # Now handled by edit_ghost_config
        params_to_edit.pop('geometric', None)  # Already handled above
        
        for key, value in params_to_edit.items():
            if key in ['num_parameters', 'dataset_config_name', 'training_mode']:
                continue
            
            new_value_str = input(f"Enter value for '{key}' (current: {value}): ")
            if new_value_str:
                original_type = type(value)
                if original_type == bool:
                    params_to_edit[key] = new_value_str.lower() in ['true', 't', '1', 'yes', 'y']
                else:
                    params_to_edit[key] = original_type(new_value_str)
        
        # Update config with changes
        for key, value in params_to_edit.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to continue...")
        return config


def edit_geometric_config(config: MoEConfig):
    """Sub-menu for geometric training configuration."""
    
    print("\n--- Geometric Training Configuration ---")
    
    if config.training_mode == 'geometric':
        print("\nGeometric Training Parameters:")
        
        # Core geometric parameters
        geom_lr = input(f"Geometric learning rate [{config.geometric.geometric_learning_rate}]: ")
        if geom_lr:
            config.geometric.geometric_learning_rate = float(geom_lr)
        
        expert_lr = input(f"Expert learning rate [{config.geometric.expert_learning_rate}]: ")
        if expert_lr:
            config.geometric.expert_learning_rate = float(expert_lr)
        
        rotation_dims = input(f"Rotation dimensions [{config.geometric.rotation_dimensions}]: ")
        if rotation_dims:
            config.geometric.rotation_dimensions = int(rotation_dims)
        
        # Loss weights
        print("\nLoss Weights:")
        ortho_weight = input(f"Orthogonality weight [{config.geometric.orthogonality_weight}]: ")
        if ortho_weight:
            config.geometric.orthogonality_weight = float(ortho_weight)
        
        rot_eff_weight = input(f"Rotation efficiency weight [{config.geometric.rotation_efficiency_weight}]: ")
        if rot_eff_weight:
            config.geometric.rotation_efficiency_weight = float(rot_eff_weight)
        
        spec_weight = input(f"Specialization weight [{config.geometric.specialization_weight}]: ")
        if spec_weight:
            config.geometric.specialization_weight = float(spec_weight)
        
        # Lambda calculus specific
        print("\nLambda Calculus Options:")
        lambda_cog = input(f"Lambda cognitive rotations [true/false] [{config.geometric.lambda_cognitive_rotations}]: ")
        if lambda_cog.lower() in ['true', 'false']:
            config.geometric.lambda_cognitive_rotations = lambda_cog.lower() == 'true'
        
        lambda_sched = input(f"Lambda rotation scheduling [curriculum/adaptive/fixed] [{config.geometric.lambda_rotation_scheduling}]: ")
        if lambda_sched in ['curriculum', 'adaptive', 'fixed']:
            config.geometric.lambda_rotation_scheduling = lambda_sched
    
    return config

def edit_ghost_config(config: MoEConfig):
    """Sub-menu for ghost expert configuration."""
    
    print("\n--- Ghost Expert Configuration ---")
    
    if config.ghost.num_ghost_experts > 0:
        print("\nGhost Expert Parameters:")
        
        # Core ghost parameters
        activation_thresh = input(f"Ghost activation threshold [{config.ghost.ghost_activation_threshold}]: ")
        if activation_thresh:
            config.ghost.ghost_activation_threshold = float(activation_thresh)
        
        ghost_lr = input(f"Ghost learning rate [{config.ghost.ghost_learning_rate}]: ")
        if ghost_lr:
            config.ghost.ghost_learning_rate = float(ghost_lr)
        
        activation_sched = input(f"Ghost activation schedule [gradual/binary/selective] [{config.ghost.ghost_activation_schedule}]: ")
        if activation_sched in ['gradual', 'binary', 'selective']:
            config.ghost.ghost_activation_schedule = activation_sched
        
        # Monitoring and coupling
        print("\nMonitoring & Learning Rate Coupling:")
        monitoring_window = input(f"Saturation monitoring window [{config.ghost.saturation_monitoring_window}]: ")
        if monitoring_window:
            config.ghost.saturation_monitoring_window = int(monitoring_window)
        
        lr_coupling = input(f"Ghost LR coupling [inverse/complementary] [{config.ghost.ghost_lr_coupling}]: ")
        if lr_coupling in ['inverse', 'complementary']:
            config.ghost.ghost_lr_coupling = lr_coupling
        
        background_learning = input(f"Ghost background learning [true/false] [{config.ghost.ghost_background_learning}]: ")
        if background_learning.lower() in ['true', 'false']:
            config.ghost.ghost_background_learning = background_learning.lower() == 'true'
    else:
        print("\nNo ghost experts configured. Set num_ghost_experts > 0 to enable ghost parameters.")
    
    return config

def train_new_model_menu(existing_config: MoEConfig = None):
    """
    A wizard for configuring and launching a new training run.
    Can be used for a new model or to evolve an existing one.
    """
    if existing_config:
        config = existing_config
        config.run_name = f"{config.run_name}_v2" # Suggest a new version name
        print(color_text("Loaded existing config. Evolving model...", Colors.YELLOW))
    else:
        config = MoEConfig()

    while True:
        clear_screen()
        print_header("Train New Model > Configuration Wizard")
        
        # Display current configuration with color coding
        print("\n--- Current Configuration ---")
        print(f" 1. Architecture       : {config.architecture_mode}")
        print(f" 2. Run Name           : {config.run_name}")
        print(f" 3. Batch Size         : {color_text(config.batch_size, Colors.GREEN)}")
        print(f" 4. Num Experts        : {color_text(config.num_experts, Colors.GREEN)}")
        
        ghost_color = Colors.GREEN if config.ghost.num_ghost_experts > 0 else Colors.RED
        print(f" 5. Num Ghost Experts  : {color_text(config.ghost.num_ghost_experts, ghost_color)}")
        
        print(f" 6. Dataset            : {config.dataset_source} -> {config.dataset_name}")
        print(f" 7. Training Mode      : {color_text(config.training_mode, Colors.YELLOW)}")
        print(f" 8. Embed Dimension    : {color_text(config.embed_dim, Colors.GREEN)}")

        is_geometric = config.training_mode == 'geometric'
        geom_color = Colors.GREEN if is_geometric else Colors.RED
        geom_dims_text = config.geometric.rotation_dimensions if is_geometric else "N/A"
        print(f" 9. Geometric Dims     : {color_text(geom_dims_text, geom_color)}")

        print(f"10. Advanced Config... : (Edit all other parameters)")
        print("-" * 29)

        print("\n[S] Start Training")
        print("[E] Exit to Main Menu")
        
        choice = input("\nEnter a number to edit, or a command (S/E): ").lower()

        try:
            if choice == '1':
                modes = ['gnn', 'hgnn', 'orthogonal', 'ghost']
                mode_choice = input(f"Enter architecture [{', '.join(modes)}] [{config.architecture_mode}]: ")
                if mode_choice in modes: 
                    config.architecture_mode = mode_choice
            elif choice == '2':
                config.run_name = input(f"Enter run name [{config.run_name}]: ") or config.run_name
            elif choice == '3':
                config.batch_size = int(input(f"Enter batch size [{config.batch_size}]: ") or config.batch_size)
            elif choice == '4':
                config.num_experts = int(input(f"Enter num experts [{config.num_experts}]: ") or config.num_experts)
            elif choice == '5':
                num_ghosts = input(f"Enter num ghost experts [{config.ghost.num_ghost_experts}]: ")
                config.ghost.num_ghost_experts = int(num_ghosts) if num_ghosts else config.ghost.num_ghost_experts
            elif choice == '6':
                config = edit_dataset_menu(config)
            elif choice == '7':
                modes = ['standard', 'geometric']
                mode_choice = input(f"Enter training mode [{', '.join(modes)}] [{config.training_mode}]: ")
                if mode_choice in modes:
                    config.training_mode = mode_choice
            elif choice == '8':
                config.embed_dim = int(input(f"Enter embed dimension [{config.embed_dim}]: ") or config.embed_dim)
            elif choice == '9':
                if is_geometric:
                    dims = input(f"Enter geometric rotation dimensions [{config.geometric.rotation_dimensions}]: ")
                    config.geometric.rotation_dimensions = int(dims) if dims else config.geometric.rotation_dimensions
                else:
                    print(color_text("\nThis option is only available in 'geometric' training mode.", Colors.RED))
                    input("Press Enter to continue...")
            elif choice == '10':
                config = advanced_config_menu(config)
            elif choice == 's':
                print("\nFinalizing configuration...")
                config.__post_init__()
                
                # Create the run-specific checkpoint directory
                config.checkpoint_dir = os.path.join("checkpoints", config.run_name)
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                
                print(f"Using device: {device}")
                print("Instantiating model...")
                
                new_model = MoEModel(config).to(device)
                
                # If evolving, transfer weights from the old model
                if existing_config and state["model"]:
                    print("Transferring weights from old model...")
                    new_model.load_state_dict(state["model"].state_dict(), strict=False)
                
                config.num_parameters = new_model.get_total_params()
                print(f"✅ Model initialized with {config.num_parameters/1e6:.2f}M parameters.")

                optimizer = create_dynamic_optimizer(new_model, config)
                scheduler = PrimaryGhostLRScheduler(config, optimizer)
                
                state.update({
                    "config": config, "model": new_model, "optimizer": optimizer,
                    "scheduler": scheduler, "tokenizer": tokenizer, "device": device,
                    "checkpoint_path": os.path.join(config.checkpoint_dir, "checkpoint.pt"),
                    "start_epoch": 0, "resume_step": 0, "best_eval_loss": float('inf')
                })
                
                launch_training()
                return
            elif choice == 'e':
                return
        except ValueError:
            print("\n❌ Invalid input. Please enter a valid number.")
            input("Press Enter to continue...")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            input("Press Enter to continue...")

def main_menu():
    """The main menu of the application."""
    while True:
        print_header("Main Menu")
        print_model_status()
        print("1. Train New Model")
        print("2. Load Model from Checkpoint")
        print("3. Continue Training / Evolve Model")
        print("4. Run Inference")
        print("5. Generate Analysis Plots")
        print("6. Exit")
        
        choice = input("> ")
        
        if choice == '1':
            train_new_model_menu()
        elif choice == '2':
            load_model_menu()
        elif choice == '3':
            if not state["model"]:
                print("\nNo model loaded. Please load a model first.")
                input("Press Enter to continue...")
                load_model_menu()
            else:
                train_new_model_menu(existing_config=state["config"])
        elif choice == '4':
            if not state["model"]:
                print("\nNo model loaded. Please load a model first.")
                input("Press Enter to continue...")
                load_model_menu()
            else:
                run_inference_menu()
        elif choice == '5':
            if not state["model"]:
                print("\nNo model loaded. Please load a model first.")
                input("Press Enter to continue...")
                load_model_menu()
            else:
                generate_plots_menu()
        elif choice == '6':
            break
        else:
            print("Invalid choice, please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main_menu()
