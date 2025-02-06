import os
from torchvision import transforms
from PIL import Image

def augment_images(input_dir, output_dir, num_augmented=200):
    # Define augmentation transformations
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.2)),  # Random cropping and scaling
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness/contrast adjustments
        transforms.RandomRotation(10),  # Random rotation (±10 degrees)
    ])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Apply augmentation to each image
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path)

        for i in range(num_augmented):
            augmented_image = augmentation(image)
            augmented_image.save(os.path.join(output_dir, f"{image_name.split('.')[0]}_aug_{i}.jpg"))

# Example usage
input_dir = "/app/data/provisional/products/product_2/images/"
output_dir = "/app/data/provisional/products/product_2/augmented/"
augment_images(input_dir, output_dir)