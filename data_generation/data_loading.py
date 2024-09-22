import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomDataset(Dataset):
    def __init__(self, captions_file, image_dir, transform=None):
        if captions_file.endswith('.json'):
            with open(captions_file, 'r') as f:
                self.captions = json.load(f)
            self.image_names = list(self.captions.keys())
        else:
            raise ValueError("Unsupported file format. Only JSON is supported.")
        
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_label = self.captions.get(img_name)

        if not img_label or 'image' not in img_label or 'caption' not in img_label:
            return None

        image_name = img_label['image'].strip()
        text = img_label['caption']

        image_path = os.path.join(self.image_dir, image_name)

        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, text

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    
    return images, list(texts)
