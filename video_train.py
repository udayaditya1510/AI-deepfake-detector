import torch
import torch.nn as nn
from video_model import VideoDeepfakeDetector
from video_utils import extract_faces
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoDeepfakeDetector().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def load_video(video_path):
    faces = extract_faces(video_path)
    faces = np.array(faces) / 255.0
    faces = torch.tensor(faces).permute(0,3,1,2)
    return faces.unsqueeze(0)

for label, cls in enumerate(["real", "fake"]):
    folder = f"video_dataset/{cls}"
    for video in os.listdir(folder):
        video_path = os.path.join(folder, video)
        inputs = load_video(video_path).to(device)
        target = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(video, loss.item())

torch.save(model.state_dict(), "video_deepfake_model.pth")
