import torch
from torchvision import models
from utils.dog_breeds import dog_breeds
from utils.preprocess import preprocess_image

# Load pre-trained model (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set model to evaluation mode

# Load ImageNet classes
imagenet_classes = {idx: entry.strip() for (idx, entry) in enumerate(open("data/imagenet_classes.txt"))}

def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted_idx = outputs.max(1)
    predicted_class = imagenet_classes[predicted_idx.item()]

    if predicted_class in dog_breeds:
        return f"✅ {predicted_class}"
    else:
        return f"❌ This is NOT a dog."
