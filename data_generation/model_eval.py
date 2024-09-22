import torch
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(r"paste data path")

from text_to_image_model import TextToImageModel
from data_loading import CustomDataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextToImageModel().to(device)

model_weights_path = r"paste model path"
model.load_state_dict(torch.load(model_weights_path))

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_directory = r"images path"
deep_fashion_captions = r"labels path"
generated_data = r"labels path"
synthetic_output_dir = r"images path"

deep_fashion_train = CustomDataset(captions_file=deep_fashion_captions, image_dir=image_directory, transform=transform)
synthetic_dataset = CustomDataset(captions_file=os.path.join(synthetic_output_dir, "captions.json"), image_dir=synthetic_output_dir, transform=transform)

combined_dataset = ConcatDataset([deep_fashion_train, synthetic_dataset])

train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

criterion = torch.nn.MSELoss()

test_image_paths = []
test_captions = []
test_losses = []

test_loss = 0.0
with torch.no_grad():
    for batch_idx, (images, texts) in enumerate(test_loader):
        if len(images) == 0 or len(texts) == 0:
            continue

        if isinstance(images, list):
            if len(images) > 0:
                images = torch.stack(images)
            else:
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
        test_loss += loss.item()

        test_losses.append(loss.item())

        if batch_idx < 5:
            test_image_paths.append(images.cpu())
            test_captions.extend(texts)

average_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {average_test_loss}")

output_data = {"captions": test_captions, "loss": test_losses}
with open("test_results.json", "w") as outfile:
    json.dump(output_data, outfile)

plt.figure(figsize=(10, 5))
plt.plot(range(len(test_losses)), test_losses, label='Batch Loss')
plt.xlabel('Batch Index')
plt.ylabel('Loss')
plt.title('Test Loss per Batch')
plt.legend()
plt.grid(True)
plt.savefig('test_loss_plot.png')
plt.show()
