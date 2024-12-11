from datasets import load_dataset
import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms

def create_difference_drawing_dataset():
    # Create output directories
    output_dir = "pipe_drawings_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "first_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "second_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "drawings"), exist_ok=True)

    # Load PIPE dataset
    print("Loading PIPE dataset...")
    dataset = load_dataset("paint-by-inpaint/PIPE", "default")
    print("Dataset loaded")

    # Process each instance
    for idx, instance in enumerate(dataset['train']):
        # Get the first and second images
        first_image = instance['first_image']
        second_image = instance['second_image']
        
        # Convert to numpy arrays
        first_array = np.array(first_image)
        second_array = np.array(second_image)
        
        # Calculate difference mask
        difference = np.abs(first_array - second_array)
        difference_mask = (difference.sum(axis=2) > 0).astype(np.uint8) * 255
        
        # Create drawing only in the difference region
        drawing = np.zeros_like(first_array)
        drawing[difference_mask > 0] = [0, 0, 0]  # Black drawing in difference regions
        
        # Save images
        Image.fromarray(first_array).save(
            os.path.join(output_dir, "first_images", f"image_{idx}.png"))
        Image.fromarray(second_array).save(
            os.path.join(output_dir, "second_images", f"image_{idx}.png"))
        Image.fromarray(drawing).save(
            os.path.join(output_dir, "drawings", f"drawing_{idx}.png"))
        
        if idx % 100 == 0:
            print(f"Processed {idx} images")

if __name__ == "__main__":
    create_difference_drawing_dataset()