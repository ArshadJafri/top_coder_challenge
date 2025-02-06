from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np
import torch  # Missing import for torch

def refine_mask(mask_path, expected_size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize mask based on expected size (example logic)
    if expected_size == "small":
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    elif expected_size == "large":
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def inpaint_product(background, mask, prompt, model_path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    
    result = pipe(
        prompt=prompt,
        image=background,
        mask_image=mask,
    ).images[0]
    
    return result  # Removed incorrect period `.`
