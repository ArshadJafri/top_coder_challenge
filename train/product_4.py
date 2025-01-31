from transformers import TrainingArguments
from torchvision import transforms
import torch
import os
from PIL import Image

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
])

image_dir = "/app/data/provisional/products/product_4/images/"

print(f"Found image directory: {image_dir}")

image_files = os.listdir(image_dir)

images = [Image.open(os.path.join(image_dir, f)) for f in image_files]

augmented_images = []
for img in images:
    for _ in range(200):
        augmented_images.append(train_transforms(img))

augmented_dir = "/app/data/provisional/products/product_4/augmented/"
os.makedirs(augmented_dir, exist_ok=True)

for i,img in enumerate(augmented_images):
    img.save(os.path.join(augmented_dir, f"{i}.jpg"))

print(f"Augmented images saved in: {augmented_dir}")

training_args = TrainingArguments(
    output_dir="/app/models/product_4/",
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-6,
    num_train_epochs=3,
    save_total_limit=2,
)

print("Transformers TrainingArguments Set!")    
print("Ready for training!")