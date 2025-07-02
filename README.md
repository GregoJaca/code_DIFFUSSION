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
├── classifier_model.py   # Defines the LeNet-5 model for classifying MNIST digits.
├── classify.py           # Classifies generated tensors using a pretrained model.
├── clean_input_output.sh # Deletes all files in the 'inputs' and 'outputs' directories.
├── clean_output.sh       # Deletes all files in the 'outputs' directory.
├── compare_tensors.py    # Compares all tensors in a directory to check for equality.
├── config.py             # Central configuration file for all parameters and default arguments.
├── extraction_system.py  # Core classes for model loading, state extraction, and running the denoising process.
├── generate_inputs.py    # Script to create initial noise files.
├── generate_perturbed_prompts.py # Script to create noise files which are small perturbations of one another.
├── generate_plane_prompts.py # Generates a 2D plane of noise tensors for exploring the latent space.
├── main.py               # Main script to run the entire extraction pipeline.
├── pipeline.py           # Runs the full pipeline from data generation to classification and visualization.
├── requirements.txt      # Python package dependencies.
├── tensor_to_image.py    # Converts a tensor or a sequence of tensors into PNG images.
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
-   `--extract_states` / `--no-extract_states`: Whether to perform the full hidden state extraction or only generate the final image. Defaults to `True`.

The script will log its progress, including timing and memory usage, and save the results in the `/outputs` directory with a clear naming convention.

### Step 3: Convert Tensor Sequences to Images

After running `main.py` with `--extract_states` (default behavior), the full denoising sequences are saved as `.pt` tensor files in `outputs/sequences/`. You can convert a specific step from these sequences into a PNG image using `tensor_to_image.py`.

```bash
python tensor_to_image.py --tensor-path "outputs/sequences/sample_0000_sequence.pt" --step -1
```

-   `--tensor-path`: **(Required)** Path to the input tensor file (e.g., `outputs/sequences/sample_0000_sequence.pt`).
-   `--step`: The generation step to visualize. Defaults to `-1` (the last, most refined step). You can specify any valid step number (0 to `num_steps - 1`).
-   The output image will be saved in `outputs/images/sequences/` with a name indicating the original tensor file and the chosen step (e.g., `sample_0000_sequence_step_0049.png`).

## Configuration

All key parameters can be modified in `config.py`:

-   **`SystemConfig`**: Defines system-level settings like the device (`cuda` or `cpu`), input/output directories, and the global random seed. It also includes a list of colors for classification visualization.
-   **`GenerationConfig`**: Holds parameters for generating initial noise tensors, such as the number of samples, image dimensions, and input channels.
-   **`ModelConfig`**: Specifies the Hugging Face model ID to be used and, crucially, maps this ID to the names of the model's first and last convolutional layers, allowing for model-agnostic hooking.
-   **`InferenceConfig`**: Contains settings for the denoising process, including batch size, number of inference steps, and whether to flatten hidden state outputs.
-   **`ClassifierConfig`**: Provides configuration for the classification script, such as the default path to tensors and the location of the pretrained classifier model weights.
-   **`DefaultConfig`**: Contains default arguments for various scripts, centralizing parameters for seeds, perturbation settings, and directory paths to ensure consistency and ease of modification.

Command-line arguments provided to `main.py` will override the defaults set in `config.py`.
