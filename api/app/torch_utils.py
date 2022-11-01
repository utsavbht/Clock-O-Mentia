import io
import gc

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 6)

    def forward(self, x):
        x = self.model(x)
        return x

model = CustomEfficientNet('efficientnet_b2')
model.load_state_dict(torch.load('app/model.pt', map_location=torch.device('cpu')))
model.eval()
gc.collect()

def transform_image(image_bytes):
    my_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((512, 512)),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(tensor):
    
    outputs = model(tensor)
    s = F.softmax(outputs)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    probabilities = " ".join(f"{i:.3f}" for i in s[0].tolist())
    gc.collect()

    return predicted_idx, probabilities
