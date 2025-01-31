from transformers import TrainingArguments
from torchvision import transforms
import torch
import os
from PIL import Image  # ✅ Fix: Import Image

# Augment 5 images to 1,000 using cropping/scaling
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
])

# Load images from the correct absolute path
image_dir = "/app/data/provisional/products/product_1/images/"
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Directory not found: {image_dir}")

print(f"✅ Found image directory: {image_dir}")

# List images and verify existence
image_files = os.listdir(image_dir)
if not image_files:
    raise FileNotFoundError(f"❌ No images found in {image_dir}")

images = [Image.open(os.path.join(image_dir, f)) for f in image_files]

# Apply augmentations
augmented_images = []
for img in images:
    for _ in range(200):  # 5 * 200 = 1,000 images
        augmented_images.append(train_transforms(img))

# Save augmented images to absolute path
augmented_dir = "/app/data/provisional/products/product_1/augmented/"
os.makedirs(augmented_dir, exist_ok=True)

for i, img in enumerate(augmented_images):
    img.save(os.path.join(augmented_dir, f"{i}.jpg"))

print(f"✅ Augmented images saved in: {augmented_dir}")

# Define training arguments using transformers' TrainingArguments
training_args = TrainingArguments(
    output_dir="/app/models/product_1/",  # Directory to save model
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-6,
    num_train_epochs=3,  # Adjust epochs as needed
    save_total_limit=2,  # Keeps only the latest 2 checkpoints
)

print("✅ Transformers TrainingArguments Set!")

# (Use Hugging Face’s Trainer class here)
print("🚀 Ready for training!")
