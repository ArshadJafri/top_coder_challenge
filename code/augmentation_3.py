import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

# Load product images and metadata
input_dir = "/app/data/provisional/products/product_1/augmented/"
meta = json.load(open("/app/data/provisional/products/product_1/meta.json"))

# Create dataset from augmented images
class AugmentedDataset(Dataset):
    def __init__(self, image_dir, tokenizer, prompt):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        inputs = self.tokenizer(
            self.prompt, return_tensors="pt", padding="max_length", truncation=True
        )
        return {"pixel_values": image, "input_ids": inputs.input_ids.squeeze()}

# Create dataset
train_dataset = AugmentedDataset(input_dir, tokenizer, meta["prompt"])

# Fine-tune the model using DreamBooth
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.fine_tune(train_dataset, meta["prompt"])

# Save the fine-tuned model
model.save_pretrained("/submission/models/product_1/")