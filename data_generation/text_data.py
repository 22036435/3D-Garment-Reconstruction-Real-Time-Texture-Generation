import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import pandas as pd
import os
import json
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
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
        elif captions_file.endswith('.csv'):
            df = pd.read_csv(captions_file)
            self.captions = {row['image']: {'image': row['image'], 'caption': row['caption']} for _, row in df.iterrows()}
            self.image_names = list(self.captions.keys())
        else:
            raise ValueError("Unsupported file format. Only JSON and CSV are supported.")
        
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

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

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

class SyntheticDataset(Dataset):
    def __init__(self, json_file, output_dir, transform=None):
        with open(json_file, 'r') as f:
            captions_data = json.load(f)
            self.prompts = [entry['caption'] for entry in captions_data.values()]

        self.transform = transform
        self.output_dir = output_dir
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

        os.makedirs(self.output_dir, exist_ok=True)

    def generate_image_from_prompt(self, prompt):
        with torch.no_grad():
            generated_image = self.pipeline(prompt)["images"][0]
        return generated_image

    def save_images_and_captions(self):
        captions_data = {}
        for i, prompt in enumerate(self.prompts):
            image = self.generate_image_from_prompt(prompt)
            image_name = f"synthetic_image_{i}.png"
            image_path = os.path.join(self.output_dir, image_name)

            image.save(image_path)

            captions_data[image_name] = {"image": image_name, "caption": prompt}

        with open(os.path.join(self.output_dir, "captions.json"), 'w') as f:
            json.dump(captions_data, f)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        img_name = f"synthetic_image_{idx}.png"
        img_path = os.path.join(self.output_dir, img_name)

        if not os.path.exists(img_path):
            return None

        image = Image.open(img_path).convert("RGB")
        text = self.prompts[idx]

        if self.transform:
            image = self.transform(image)

        return image, text


image_directory = r"images path"
deep_fashion_captions = r"labels path"
generated_data = r"laebls path"
synthetic_output_dir = r"images path"

synthetic_data = SyntheticDataset(json_file=generated_data, output_dir=synthetic_output_dir, transform=transform)
synthetic_data.save_images_and_captions()

deep_fashion_train = CustomDataset(captions_file=deep_fashion_captions, image_dir=image_directory, transform=transform)

synthetic_dataset = CustomDataset(captions_file=os.path.join(synthetic_output_dir, "captions.json"), image_dir=synthetic_output_dir, transform=transform)

combined_dataset = ConcatDataset([deep_fashion_train, synthetic_dataset])

train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc = nn.Linear(self.clip_model.config.projection_dim, 256*256*3)

    def forward(self, text_inputs):
        text_features = self.clip_model.get_text_features(**text_inputs)
        images = self.fc(text_features)
        images = images.view(-1, 3, 256, 256)
        return images

model = TextToImageModel().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, texts in train_loader:

        if len(images) == 0 or len(texts) == 0:
            continue

        images = images.to(device)

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError(f"Expected texts to be a list or string, but got {type(texts)}")

        text_inputs = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        outputs = model(text_inputs)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}")

model_save_path = "text_to_image_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

with open('training_loss.json', 'w') as f:
    json.dump(train_losses, f)

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()

print("Training complete.")
