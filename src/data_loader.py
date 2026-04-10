import os
import json
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path


# PyTorch Dataset for Fashionpedia outfit images
# Loads images from train/ directory (recursively)
# Loads labels from data/labels/pseudo_labels.json
# Each label has 5 attributes:
#   - formal: float 0.0-1.0 (regression)
#   - season: int 1-4 (classification)
#   - gender: int 0-2 (classification)
#   - time: int 0-1 (classification)
#   - frequency: int 0-1 (classification)
# __getitem__ returns (image_tensor, label_dict)
# Image transforms: resize 224x224, normalize with CLIP mean/std
#   mean = [0.48145466, 0.4578275, 0.40821073]
#   std  = [0.26862954, 0.26130258, 0.27577711]
# Split into train/val with 80/20 ratio
# Skip images not found in pseudo_labels.json


class FashionpediaDataset(Dataset):
    """PyTorch Dataset for Fashionpedia outfit images with multi-task attributes."""
    
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(self, image_paths, labels_dict, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels_dict: Dictionary mapping image filenames to label dictionaries
            transform: Optional torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Returns (image_tensor, label_dict)."""
        image_path = self.image_paths[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get filename for label lookup
        filename = os.path.basename(image_path)
        
        # Retrieve label dictionary
        label_dict = self.labels_dict.get(filename, {
            'formal': 0.0,
            'season': 1,
            'gender': 0,
            'time': 0,
            'frequency': 0
        })

        labels = {
            'formal':    torch.tensor(label_dict['formal'],     dtype=torch.float32),
            'season':    torch.tensor(label_dict['season'] - 1, dtype=torch.long),
            'gender':    torch.tensor(label_dict['gender'],     dtype=torch.long),
            'time':      torch.tensor(label_dict['time'],       dtype=torch.long),
            'frequency': torch.tensor(label_dict['frequency'],  dtype=torch.long),
        }

        return image_tensor, labels


def load_fashionpedia_data(data_dir, images_dir='train', labels_file='data/labels/pseudo_labels.json'):
    """
    Load Fashionpedia dataset with train/val split.
    
    Args:
        data_dir: Root directory of the project
        images_dir: Directory containing outfit images (relative to data_dir)
        labels_file: Path to pseudo_labels.json (relative to data_dir)
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Build full paths
    images_path = os.path.join(data_dir, images_dir)
    labels_path = os.path.join(data_dir, labels_file)
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    # Recursively find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_paths = []
    
    for root, dirs, files in os.walk(images_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                full_path = os.path.join(root, file)
                # Only include images that have labels
                if file in labels_dict:
                    image_paths.append(full_path)
    
    # Create dataset
    dataset = FashionpediaDataset(image_paths, labels_dict)
    
    # Split into train/val with 80/20 ratio
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset
