# extraction_system.py
"""Core components for model loading, state extraction, and processing."""
import os
import torch
import logging
from typing import List, Dict, Tuple
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForwardHook:
    """A hook to capture the output of a specific module."""
    def __init__(self):
        self.storage: List[torch.Tensor] = []

    def __call__(self, module, input, output):
        # Detach from graph and move to CPU to save GPU memory,
        # but only if not processing immediately on GPU.
        # For this task, we process after the loop, so we keep it on GPU for now.
        self.storage.append(output)

    def clear(self):
        """Clears the stored tensors."""
        self.storage.clear()

class StateExtractor:
    """Manages the extraction of hidden states from a diffusion model."""

    def __init__(self, model_id: str, device: str, first_layer_name: str, last_layer_name: str):
        self.device = device
        self.model_id = model_id
        
        logging.info(f"Loading model '{model_id}'...")
        try:
            self.model = UNet2DModel.from_pretrained(model_id).to(self.device)
            self.scheduler = DDPMScheduler.from_pretrained(model_id)
        except Exception as e:
            logging.error(f"Failed to load model '{model_id}'. Ensure it's a UNet2DModel. Error: {e}")
            raise
            
        self.first_layer_name = first_layer_name
        self.last_layer_name = last_layer_name

        self.hooks = {}
        self.captured_states: Dict[str, List[torch.Tensor]] = {
            "first_layer": [],
            "last_layer": []
        }
        
    def _attach_hooks(self):
        """Finds and attaches hooks to the specified model layers."""
        for name, module in self.model.named_modules():
            if name == self.first_layer_name or name == self.last_layer_name:
                hook = ForwardHook()
                handle = module.register_forward_hook(hook)
                self.hooks[name] = (hook, handle)
                logging.info(f"Attached hook to layer: {name}")

    def _remove_hooks(self):
        """Removes all attached hooks."""
        for name, (hook, handle) in self.hooks.items():
            handle.remove()
            hook.clear()
            logging.info(f"Removed hook from layer: {name}")
        self.hooks.clear()

    def run_extraction(
        self,
        initial_noise: torch.Tensor,
        num_steps: int,
        flatten_output: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Runs the full denoising process and extracts hidden states.

        Args:
            initial_noise (torch.Tensor): The starting noise tensor (x_T).
            num_steps (int): Number of denoising steps.
            flatten_output (bool): If True, flattens spatial dimensions.

        Returns:
            A tuple containing:
            - The full x_0 to x_T sequence tensor.
            - A dictionary of the extracted layer states over all timesteps.
        """
        self.scheduler.set_timesteps(num_steps)
        batch_size = initial_noise.shape[0]
        image = initial_noise.to(self.device)
        
        sequence_x = [image.cpu().clone()]

        self._attach_hooks()

        try:
            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    # Predict noise
                    noise_pred = self.model(image, t).sample
                    # Compute previous image state
                    image = self.scheduler.step(noise_pred, t, image).prev_sample
                sequence_x.append(image.cpu().clone())
        finally:
            # Ensure hooks are always removed
            first_layer_hook = self.hooks[self.first_layer_name][0]
            last_layer_hook = self.hooks[self.last_layer_name][0]
            
            # --- Process on GPU then move to CPU ---
            # Stack along a new 'timestep' dimension
            first_layer_states = torch.stack(first_layer_hook.storage, dim=1)
            last_layer_states = torch.stack(last_layer_hook.storage, dim=1)

            if flatten_output:
                # Reshape on GPU: (batch, timesteps, C, H, W) -> (batch, timesteps, C*H*W)
                first_layer_states = first_layer_states.view(batch_size, num_steps, -1)
                last_layer_states = last_layer_states.view(batch_size, num_steps, -1)

            # Move final processed tensors to CPU
            processed_states = {
                "layer1": first_layer_states.cpu(),
                "last_layer": last_layer_states.cpu()
            }
            self._remove_hooks()
        
        # Reverse sequence to be x_0, x_1, ... x_T
        full_sequence = torch.stack(list(reversed(sequence_x)), dim=1)
        
        # Clean GPU memory
        del first_layer_states, last_layer_states, image, noise_pred
        torch.cuda.empty_cache()

        return full_sequence, processed_states