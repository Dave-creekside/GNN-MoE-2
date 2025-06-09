# MoE Research Hub - User Manual

## 1. Introduction

Welcome to the User Manual for the MoE Research Hub. This guide provides detailed instructions on how to use the interactive Command-Line Interface (`app.py`) to manage the entire lifecycle of your research experiments.

This manual is a companion to the main `README.md`. For a high-level overview of the project's research goals and architectural evolution, please refer to the [README.md](README.md). This document focuses on the practical "how-to" of using the software.

### Getting Started

All functionality is accessed through the main application script. To begin, run the following command from the root of the project directory:

```bash
python3 app.py
```

---

## 2. The Interactive Application (`app.py`)

The MoE Research Hub is a menu-driven application designed to be intuitive and powerful. Below is a detailed walkthrough of each menu and its capabilities.

### 2.1 Main Menu

Upon launching the app, you are greeted with the Main Menu. This is the central navigation point.

```
============================================================
🧠 MoE Research Hub: Main Menu
============================================================

--- Model Status ---
No model loaded.
--------------------

1. Train New Model
2. Load Model from Checkpoint
3. Exit
>
```

*   **Train New Model**: This option launches the Configuration Wizard to create and train a new model from scratch.
*   **Load Model from Checkpoint**: This allows you to load a previously trained model to run inference, continue training, or perform analysis.
*   **Exit**: Closes the application.

### 2.2 Training a New Model

Selecting `1. Train New Model` starts the Configuration Wizard. This is a powerful tool for crafting your experiments.

#### The Configuration Wizard

The wizard allows you to define every aspect of your training run. It is designed with a "simple by default, powerful when needed" philosophy.

```
Current Configuration:
 1. Architecture: ghost
 2. Run Name: moe_run
 3. Batch Size: 8
 4. Num Experts: 4
 5. Num Ghost Experts: 2
 6. Dataset: huggingface -> wikitext
 7. Advanced Configuration...

[S] Start Training with these settings
[E] Exit to Main Menu
```

*   **Simple Configuration (Options 1-6)**: You can quickly edit the most common and impactful parameters by selecting the corresponding number.
*   **Advanced Configuration (Option 7)**: For full control, this option opens a sub-menu where you can edit **every single parameter** in the `MoEConfig`, `HGNNParams`, and `GhostParams` dataclasses. This is ideal for fine-grained experimentation.
*   **Start Training (`S`)**: Once you are satisfied with your configuration, this command will initialize all components and launch the training session.
*   **Exit (`E`)**: Returns to the Main Menu without saving changes.

#### Selecting a Dataset (Option 6)

This sub-menu allows you to specify the data source for your experiment.

```
Select Dataset Source:
1. Hugging Face Hub
2. Local File (.txt, .json, .jsonl)
>
```

*   **Hugging Face Hub**: Prompts for a dataset path in the format `dataset_name/config_name` (e.g., `wikitext/wikitext-2-v1`).
*   **Local File**: Prompts for the full path to a local file. The loader can handle `.txt`, `.json`, and `.jsonl` files automatically. See the **Dataset Guide** in Section 5 for formatting details.

### 2.3 Loading an Existing Model

Selecting `2. Load Model from Checkpoint` from the Main Menu allows you to load a saved model.

*   **Prompt**: `Enter the full path to the checkpoint.pt file (e.g., checkpoints/my_run/checkpoint.pt):`
*   **Functionality**: This loads not only the model weights but also the full training state, including the optimizer, scheduler, and the exact configuration used for the run.

### 2.4 The Model Menu

After successfully loading a model, you are taken to the Model Menu, which provides contextual actions for the loaded model.

```
--- Model Status ---
Loaded Model: my_lambda_calculus_run
Architecture: ghost
Parameters: 1.98M
Checkpoint: checkpoints/my_lambda_calculus_run/checkpoint.pt
--------------------

1. Run Inference
2. Continue Training
3. View Full Configuration
4. Generate Analysis Plots
5. Return to Main Menu
>
```

*   **Run Inference**: Opens a sub-menu where you can provide a text prompt and generation parameters (`max_length`, `temperature`, `top_k`) to see your model in action.
*   **Continue Training**: Resumes the training process from the exact state saved in the checkpoint. It will first ask you to confirm or change the total number of epochs you want to run to.
*   **View Full Configuration**: Displays a clean, sectioned view of every hyperparameter used for the loaded model's training run.
*   **Generate Analysis Plots**: Re-runs the analysis script on the `training_log.json` associated with the loaded checkpoint, generating a full suite of performance plots in the checkpoint directory.
*   **Return to Main Menu**: Unloads the current model and returns you to the main menu.

---

## 3. A Deep Dive into the Architectures

The `architecture_mode` parameter in the configuration wizard is the most important setting for defining your experiment. It controls which combination of modules and loss functions are active. Here is a breakdown of each mode:

*   **`gnn`**: This is the simplest architecture. It uses a standard Transformer block for each expert but does not enable any communication between them. It serves as a baseline to measure the benefit of more complex coordination strategies.
*   **`hgnn`**: This mode activates the `HGNNExpertCoupler` module. After each expert processes the input, their outputs are fed into a Hypergraph Neural Network. The HGNN allows the experts to exchange and refine their representations based on learned group relationships before their final outputs are combined. This is the first step towards collaborative expert processing.
*   **`orthogonal`**: This mode builds directly on `hgnn` by adding an **orthogonality loss** to the training objective. This loss function actively encourages the weight matrices of the different experts to be dissimilar, or "orthogonal." This is a powerful technique to prevent "expert collapse"—a common problem where all experts learn to perform the same function—and ensures a diverse and specialized set of experts.
*   **`ghost`**: This is the most advanced architecture, combining all previous features with the **Ghost Expert** mechanism. It uses HGNN coupling and orthogonality loss, but also includes a secondary pool of "ghost" experts. These experts are dormant by default but are dynamically activated if the model detects that the primary experts are "saturating" (i.e., becoming too similar and failing to explain the variance in the data). This gives the model adaptive capacity, allowing it to scale its complexity on demand for more challenging inputs.

---

## 4. Hyperparameter Glossary

The following is a reference for all available parameters in the "Advanced Configuration" menu of the training wizard.

### Core Parameters
*   `run_name` (str): The name for your experiment run. Checkpoints and logs will be saved in `checkpoints/[run_name]/`.
*   `architecture_mode` (str): Selects the architecture to use. See Section 3 for details.
*   `seed` (int): The random seed for reproducibility.

### Model Architecture
*   `embed_dim` (int): The dimensionality of the token embeddings and the hidden states of the model.
*   `num_layers` (int): The number of Transformer layers in the model.
*   `num_heads` (int): The number of attention heads in the multi-head attention mechanism.
*   `num_experts` (int): The number of primary experts in each MoE layer.
*   `max_seq_length` (int): The maximum sequence length the model can process.
*   `vocab_size` (int): The size of the tokenizer's vocabulary. This is typically set automatically.
*   `dropout_rate` (float): The dropout rate applied to various layers for regularization.

### HGNN Parameters (`hgnn`)
*   `num_layers` (int): The number of layers in the HGNN coupler.
*   `strategy` (str): The method for creating the static hypergraph edges (`all_pairs`, `all_triplets`, `all`).
*   `learnable_edge_weights` (bool): If `True`, the weights of the hyperedges will be learned during training.

### Ghost Expert Parameters (`ghost`)
*   `num_ghost_experts` (int): The number of dormant ghost experts to include in each MoE layer. Setting to `0` disables the ghost mechanism.
*   `ghost_activation_threshold` (float): The saturation level at which ghost experts begin to activate.
*   `ghost_learning_rate` (float): The learning rate specifically for the ghost experts.
*   `ghost_activation_schedule` (str): The method for increasing ghost expert activation (`gradual`, `binary`).
*   `saturation_monitoring_window` (int): The number of steps over which to average expert saturation metrics.
*   `ghost_lr_coupling` (str): The method for coupling the ghost learning rate to the primary learning rate schedule (`inverse`, `complementary`).
*   `ghost_background_learning` (bool): If `True`, ghost experts will continue to learn at a very low rate even when inactive.

### Training Parameters
*   `training_loop` (str): The training loop function to use. Currently, only `"standard"` is implemented.
*   `epochs` (int): The total number of epochs to train for.
*   `batch_size` (int): The number of sequences in each training batch.
*   `learning_rate` (float): The initial learning rate for the primary model parameters.
*   `max_batches_per_epoch` (int): The maximum number of batches to process per epoch. Set to `-1` to use the full dataset.
*   `eval_every` (int): The number of training steps between each evaluation phase.

### Dataset Parameters
*   `dataset_source` (str): The source of the data (`huggingface` or `local_file`).
*   `dataset_name` (str): The path or name of the dataset.
*   `dataset_config_name` (str): The specific configuration for a Hugging Face dataset (e.g., `wikitext-2-v1`).
*   `num_train_samples` (int): The number of samples to use for training. Set to `-1` to use all available samples.
*   `num_eval_samples` (int): The number of samples to use for evaluation. Set to `-1` to use all available samples.

---

## 5. A Guide to Datasets

The framework supports loading data from both the Hugging Face Hub and local files.

### Hugging Face Datasets
*   **How to Use**: In the dataset selection menu, choose "Hugging Face Hub" and provide the dataset path in the format `dataset_name/config_name`.
*   **Example**: To use the WikiText-2 dataset, you would enter `wikitext/wikitext-2-v1`.

### Local Datasets
The framework can load local text data from `.txt`, `.json`, or `.jsonl` files. The data loader automatically splits the data into a 90% training set and a 10% validation set.

*   **`.txt` Format**:
    *   **Structure**: A plain text file with one document, paragraph, or sentence per line.
    *   **Example (`my_data.txt`)**:
        ```
        This is the first document. It contains valuable text.
        This is the second document, which will be a separate sample.
        And a third.
        ```

*   **`.json` or `.jsonl` Format**:
    *   **Structure**: The file should contain a list of JSON objects. The loader will look for a specific key to extract the text. It prioritizes the key `"text"`, but if it's not found, it will look for `"question"`, `"reasoning"`, and `"answer"` keys and combine them, which is ideal for instruction-style datasets.
    *   **Example (`my_data.jsonl`)**:
        ```json
        {"text": "This is the first document."}
        {"text": "This is the second document."}
        ```
    *   **Example (GRPO/Instruction Format)**:
        ```json
        {
          "question": "((\u03bbx.(\u03bby.(x y))) a) b",
          "reasoning": "Apply outer function...",
          "answer": "a b"
        }
        ```
        In this case, the loader will combine the values into a single text sample: `Question: ((\u03bbx...))\nReasoning: Apply outer...\nAnswer: a b`.
