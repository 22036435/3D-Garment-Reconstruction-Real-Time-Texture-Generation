import torch.nn as nn
from transformers import CLIPModel

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc = nn.Linear(self.clip_model.config.projection_dim, 256 * 256 * 3)

    def forward(self, text_inputs):
        text_features = self.clip_model.get_text_features(**text_inputs)
        images = self.fc(text_features)
        images = images.view(-1, 3, 256, 256)
        return images
