# config.py
"""Centralized configuration for the hidden state extraction system."""
from dataclasses import dataclass
import torch

@dataclass
class SystemConfig:
    """General system and I/O configurations."""
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_DIR: str = "inputs"
    OUTPUT_DIR: str = "outputs"
    SEED: int = 42

@dataclass
class GenerationConfig:
    """Configuration for input noise generation."""
    NUM_SAMPLES: int = 2
    # IMAGE_SIZE: int = 32
    # IN_CHANNELS: int = 3
    IMAGE_SIZE: int = 28
    IN_CHANNELS: int = 1

@dataclass
class ModelConfig:
    """Configuration for the diffusion model."""
    # --- Selected Model ---
    # Switch between model IDs here. Layer names are mapped below.
    MODEL_ID: str = "1aurent/ddpm-mnist"
    
    # --- Model-specific layer names for hooking ---
    # This allows the system to be agnostic to the model architecture.
    # Add new models and their layer names here.
    from dataclasses import field
    MODEL_LAYER_MAP: dict = field(default_factory=lambda: {
        "1aurent/ddpm-mnist": { 
            "first_layer": "conv_in",
            "last_layer": "conv_out"
        },
        "bot66/MNISTDiffusion": {
            "first_layer": "conv_in",
            "last_layer": "conv_out"
        },
        "google/ddpm-cifar10-32": { # Example for a different dataset
             "first_layer": "conv_in",
             "last_layer": "conv_out"
        }
    })

    @property
    def FIRST_LAYER_NAME(self) -> str:
        return self.MODEL_LAYER_MAP[self.MODEL_ID]["first_layer"]

    @property
    def LAST_LAYER_NAME(self) -> str:
        return self.MODEL_LAYER_MAP[self.MODEL_ID]["last_layer"]

@dataclass
class InferenceConfig:
    """Configuration for the denoising inference process."""
    BATCH_SIZE: int = 2
    NUM_INFERENCE_STEPS: int = 50
    GUIDANCE_SCALE: float = 7.5 # For guided diffusion, not used in DDPM
    FLATTEN_OUTPUT: bool = True # Flatten spatial dims to vectors
    EXTRACT_HIDDEN_STATES: bool = True # Whether to extract hidden states or just generate the final image

# Instantiate configurations for easy import
system_config = SystemConfig()
generation_config = GenerationConfig()
model_config = ModelConfig()
inference_config = InferenceConfig()



# For the classifier
class ClassifierConfig:
    """
    Configuration class for the MNIST classifier.
    """
    DEFAULT_TENSORS_PATH = "outputs/final_tensors"
    DEFAULT_JSON_NAME = "predictions.json"
    weights_url = "https://media.githubusercontent.com/media/a-martyn/mnist-digits-recognition-pytorch/main/model.pth"
