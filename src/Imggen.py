# Title: Image Generation and Object Replacement using Stable Diffusion 3.5
# Author: [Your Name]
# Date: December 2024

# Import necessary libraries

import torch
import cv2
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from transformers import pipeline
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download and load the SAM model for automatic mask generation
def download_sam_model():
    """
    Download the SAM model if it doesn't exist
    """
    import os
    
    filename = "sam_vit_h_4b8939.pth"
    if not os.path.exists(filename):
        print("Downloading SAM model...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download completed!")
    else:
        print("SAM model already exists.")

def load_sam_model():
    """
    Load the Segment Anything Model (SAM) for automatic mask generation
    """
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

# Load the quantized Stable Diffusion model
def load_sd_model():
    """
    Load and configure the Stable Diffusion pipeline with model quantization
    """
    model_id = "runwayml/stable-diffusion-inpainting"
    
    pipeline = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipeline.to(device)
    return pipeline

def generate_mask(image, mask_generator, target_object):
    """
    Generate automatic mask for the target object in the image
    """
    # Generate all possible masks
    masks = mask_generator.generate(image)
    
    # Convert masks to binary format with additional logic for person detection
    binary_masks = []
    for mask in masks:
        # Create binary mask
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Check if this mask is likely to be a person based on aspect ratio and position
        bbox = mask['bbox']  # [x, y, width, height]
        aspect_ratio = bbox[3] / bbox[2]  # height/width
        is_vertical = aspect_ratio > 1.2  # People are usually taller than wide
        
        # Check if mask is in reasonable size range (not too small or too large)
        area_ratio = mask['area'] / (image.shape[0] * image.shape[1])
        reasonable_size = 0.05 < area_ratio < 0.7
        
        if is_vertical and reasonable_size:
            binary_mask[mask['segmentation']] = 255
            binary_masks.append({
                'mask': binary_mask,
                'score': mask['predicted_iou'] * (1 if is_vertical else 0.5),
                'area': mask['area']
            })
    
    # If no suitable masks found, fallback to largest mask
    if not binary_masks:
        return generate_fallback_mask(masks, image.shape[:2])
    
    # Return the mask with highest score
    return max(binary_masks, key=lambda x: x['score'])['mask']

def generate_fallback_mask(masks, shape):
    """Fallback to basic mask generation"""
    binary_mask = np.zeros(shape, dtype=np.uint8)
    largest_mask = max(masks, key=lambda x: x['area'])
    binary_mask[largest_mask['segmentation']] = 255
    return binary_mask

def replace_object(image_path, target_object, replacement_prompt, sd_pipeline, mask_generator):
    """
    Replace an object in the image with a generated one
    
    Args:
        image_path: Path to input image
        target_object: Description of object to replace
        replacement_prompt: Prompt for generating replacement
        sd_pipeline: Stable Diffusion pipeline
        mask_generator: SAM mask generator
    
    Returns:
        Original image, mask, and result image
    """
    # Load and preprocess image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Convert to numpy array for mask generation
    image_np = np.array(image)
    
    # Generate mask
    mask = generate_mask(image_np, mask_generator, target_object)
    
    # Prepare mask for SD pipeline
    mask_image = Image.fromarray(mask)
    
    # Generate replacement
    result = sd_pipeline(
        prompt=replacement_prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt="low quality, blurry, distorted",
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return image, mask_image, result

def display_results(original, mask, result):
    """
    Display the original image, mask, and result side by side
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Generated Mask')
    ax2.axis('off')
    
    ax3.imshow(result)
    ax3.set_title('Result')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # Download and load models
    print("Setting up models...")
    download_sam_model()  # Add this line
    
    print("Loading models...")
    mask_generator = load_sam_model()
    sd_pipeline = load_sd_model()
    
    # Rest of your code remains the same
    image_url =  r"D:\Stable diffusion\akhacouple.webp"
    target_object = "person"
    replacement_prompt = "Two people wearing spacesuits standing in a green mountainous landscape, high quality, detailed, realistic"
    
    original, mask, result = replace_object(
        image_url,
        target_object,
        replacement_prompt,
        sd_pipeline,
        mask_generator
    )
    
    display_results(original, mask, result)

if __name__ == "__main__":
    main()