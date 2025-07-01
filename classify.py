# classify.py

import torch
import os
import json
import argparse
from torchvision import transforms
# from PIL import Image
from tqdm import tqdm

from config import ClassifierConfig


import torch
from classifier_model import LeNetMNIST # MNISTClassifier

def get_model():
    """
    Loads the LeNetMNIST classifier and attaches pre-trained weights from a local file.
    """
    model = LeNetMNIST()
    
    # Load the state dictionary from the local file
    state_dict = torch.load(ClassifierConfig.local_weights_path)

    model.load_state_dict(state_dict)
    
    return model

def classify_tensors(tensors_path, model):
    """
    Classifies a directory of tensors.

    Args:
        tensors_path (str): The path to the directory containing the .pt files.
        model (torch.nn.Module): The classification model.

    Returns:
        dict: A dictionary with filenames as keys and predicted labels as values.
    """
    predictions = {}
    model.eval()
    transform = transforms.Compose([ # GGG this is not being used, should i delete?
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    for filename in tqdm(os.listdir(tensors_path)):
        if filename.endswith(".pt"):
            file_path = os.path.join(tensors_path, filename)
            try:
                # Load the tensor and take the last generation step
                tensor = torch.load(file_path)[-1]

                # Ensure the tensor is in the correct format (1, 28, 28)
                if tensor.ndim == 3 and tensor.shape[0] == 1:
                    image_tensor = tensor
                elif tensor.ndim == 2:
                    image_tensor = tensor.unsqueeze(0)
                else:
                    print(f"Skipping {filename} due to unexpected tensor shape: {tensor.shape}")
                    continue

                with torch.no_grad():
                    output = model(image_tensor.unsqueeze(0)) # Add batch dimension
                    pred = output.argmax(dim=1, keepdim=True)
                    predictions[filename] = pred.item()

            except Exception as e:
                print(f"Could not process {filename}: {e}")

    return predictions

def main():
    """
    Main function to run the classification script.
    """
    parser = argparse.ArgumentParser(description="Classify MNIST tensors.")
    parser.add_argument(
        "--tensors_path",
        type=str,
        default=ClassifierConfig.DEFAULT_TENSORS_PATH,
        help="Path to the directory containing the .pt tensor files.",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default=ClassifierConfig.DEFAULT_JSON_NAME,
        help="Name of the output JSON file.",
    )
    args = parser.parse_args()

    model = get_model()
    predictions = classify_tensors(args.tensors_path, model)

    output_path = os.path.join(args.tensors_path, args.json_name)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()