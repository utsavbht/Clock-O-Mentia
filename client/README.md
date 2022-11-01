# Clock o' Mentia

## Description

## Android Application

## Model
EfficientNet B2 was trained on the labelled dataset using transfer learning. The last layer was modified to output a vector of length 6.
```python
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
```
## Flask API

The API is hosted [here](https://clockomentia.herokuapp.com). It returns the json response containing the predicition and the probabilities corresponding to all 6 classes.
```python
import requests

url = "https://clockomentia.herokuapp.com/predict"
response = requests.post(url, files={'file': open('test.jpg', 'rb')})
print(response.text)
```

## Dataset

NHATS_R10_ClockDrawings