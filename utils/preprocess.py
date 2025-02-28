import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"Error: Unable to download image. HTTP Status Code: {response.status_code}")

            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        return transform(image).unsqueeze(0)  # Add batch dimension

    except UnidentifiedImageError:
        raise ValueError("Error: The downloaded file is not a valid image.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error: Failed to download the image. {e}")
