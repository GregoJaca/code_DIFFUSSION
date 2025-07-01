# Diffusion Model Hidden State Extraction System

This project provides a modular and configurable Python system for extracting internal hidden states from pretrained diffusion models during the denoising inference process. It is designed for efficiency, especially on GPU, and saves the extracted data for further analysis, such as in chaos theory research.

## Features

- **Modular Architecture**: Code is separated into configuration, input generation, and core extraction logic.
- **Configurable**: Easily change models, inference parameters, and output formats via a central `config.py` file.
- **Hidden State Extraction**: Uses PyTorch hooks to capture outputs from any specified layer (defaults to the first and last).
- **Full Sequence Capture**: Saves the entire denoising trajectory from noise to image ($x_T \rightarrow x_{T-1} \rightarrow \dots \rightarrow x_0$).
- **Efficient Processing**: Performs tensor manipulations (like flattening) on the GPU before moving data to the CPU for storage, minimizing GPU memory bottlenecks.
- **Batch Processing**: Efficiently processes multiple input samples in batches.
- **Model Agnostic**: Designed to work with various `UNet2DModel`-based diffusion models from the Hugging Face Hub with minimal changes.

## Project Structure

```
/
├── inputs/               # Stores generated initial noise tensors (.pt)
├── outputs/              # Stores all extracted data
│   ├── layer1/           # Trajectories from the first specified layer
│   ├── last_layer/       # Trajectories from the last specified layer
│   └── sequences/        # Full x_T to x_0 image sequences
├── config.py             # Central configuration file for all parameters
├── generate_inputs.py    # Script to create initial noise files
├── generate_perturbed_inputs.py    # Script to create noise files which are small perturbations of one another
├── extraction_system.py  # Core classes for model loading and state extraction
├── main.py               # Main script to run the entire pipeline
├── requirements.txt      # Python package dependencies
└── README.md             # This file
```

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create Directories**:
    Create the necessary input and output directories.
    ```bash
    mkdir -p inputs outputs/layer1 outputs/last_layer outputs/sequences
    ```

3.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## How to Use

The process is a two-step pipeline: first generate inputs, then run the extraction.

### Step 1: Generate Input Noise

Run the `generate_inputs.py` script to create the initial random noise tensors that will be the starting point for the diffusion process.

```bash
python generate_inputs.py --num_samples 10
```
This will create 10 files (`noise_0000.pt`, `noise_0001.pt`, etc.) in the `/inputs` directory. You can adjust the number of samples in `config.py` or via the command-line argument.

### Step 2: Run the Extraction

Execute the main script to load the model, perform inference on the generated noise, and save the hidden states.

```bash
python main.py --batch_size 2 --num_steps 50 --flatten
```

-   `--batch_size`: Number of samples to process simultaneously. Adjust based on your GPU memory.
-   `--num_steps`: The number of timesteps in the denoising process.
-   `--flatten` / `--no-flatten`: Whether to save hidden states as flattened vectors or as spatial tensors.

The script will log its progress, including timing and memory usage, and save the results in the `/outputs` directory with a clear naming convention.

## Configuration

All key parameters can be modified in `config.py`:

-   **`system_config`**: Device (`cuda`/`cpu`), directories, and random seed.
-   **`generation_config`**: Number of samples to generate, image dimensions.
-   **`model_config`**:
    -   `MODEL_ID`: The Hugging Face Hub ID of the model to use (e.g., `"aurent/ddpm-mnist"`).
    -   `MODEL_LAYER_MAP`: **Crucial for model agnosticism**. Maps a model ID to the string names of its first and last convolutional layers. If you use a new model, inspect its architecture (`print(model)`) and add its layer names here.
-   **`inference_config`**: Default batch size, number of inference steps, and whether to flatten outputs.

Command-line arguments provided to `main.py` will override the defaults set in `config.py`.