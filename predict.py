import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "FAKE (AI Generated)" if pred.item() == 1 else "REAL"
    return label, round(conf.item() * 100, 2)


if __name__ == "__main__":
    label, confidence = predict_image("test.jpg")
    print(label, confidence)
