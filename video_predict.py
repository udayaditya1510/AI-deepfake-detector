import torch
import numpy as np
from video_model import VideoDeepfakeDetector
from video_utils import extract_faces

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VideoDeepfakeDetector().to(device)
model.load_state_dict(torch.load("video_deepfake_model.pth", map_location=device))
model.eval()

def predict_video(video_path):
    faces = extract_faces(video_path)
    faces = np.array(faces) / 255.0
    faces = torch.tensor(faces).permute(0,3,1,2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(faces)
        pred = torch.argmax(output, 1).item()

    return "FAKE VIDEO" if pred == 1 else "REAL VIDEO"
