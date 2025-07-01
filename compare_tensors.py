

import torch
import os

def compare_tensors_in_directory(directory_path):
    tensors = []
    tensor_names = []
    
    # Load all .pt files from the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pt"):
            filepath = os.path.join(directory_path, filename)
            try:
                tensor = torch.load(filepath)
                tensors.append(tensor)
                tensor_names.append(filename)
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not tensors:
        print("No .pt tensors found in the specified directory.")
        return

    unequal_pairs_count = 0
    total_pairs = 0

    # Compare all unique pairs of tensors
    num_tensors = len(tensors)
    for i in range(num_tensors):
        for j in range(i + 1, num_tensors):
            total_pairs += 1
            tensor1 = tensors[i]
            tensor2 = tensors[j]
            name1 = tensor_names[i]
            name2 = tensor_names[j]

            if not torch.equal(tensor1, tensor2):
                unequal_pairs_count += 1
                print(f"Tensors {name1} and {name2} are NOT equal.")
            else:
                print(f"Tensors {name1} and {name2} are equal.")

    print(f"\nTotal number of tensor pairs compared: {total_pairs}")
    print(f"Number of unequal tensor pairs: {unequal_pairs_count}")

if __name__ == "__main__":
    inputs_directory = "./inputs"
    compare_tensors_in_directory(inputs_directory)

