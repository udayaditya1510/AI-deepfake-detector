# 🛡️ AI Image & Deepfake Detector

An end-to-end Deepfake Detection System built using PyTorch and FastAPI that detects:

- 🖼️ AI-generated / manipulated Images
- 🎥 AI-generated / manipulated Videos

This project includes:
- CNN-based image deepfake detection
- CNN + LSTM video deepfake detection
- FastAPI backend
- Web UI support
- REST API endpoints

---

# 🚀 Project Architecture

User → FastAPI Server → Model Inference → Prediction Result

- Image detection → ResNet18 classifier  
- Video detection → ResNet18 (feature extractor) + LSTM  

---

# 🖼️ Image Deepfake Detection

## Model Architecture

Image detection uses a fine-tuned ResNet18 model:

- Pretrained on ImageNet
- Final layer modified for 2-class classification
- Output:
  - REAL
  - FAKE (AI Generated)

Files:
- model.py → Model architecture
- predict.py → Image prediction logic
- train.py → Training script

---

# 🎥 Video Deepfake Detection

## Model Architecture

Video detection pipeline:

1. Extract faces from frames using Haar Cascade
2. CNN (ResNet18) extracts spatial features
3. LSTM processes temporal sequence
4. Final classifier outputs:
   - REAL VIDEO
   - FAKE VIDEO

Files:
- video_model.py → CNN + LSTM architecture
- video_predict.py → Video prediction logic
- video_train.py → Video training script
- video_utils.py → Face extraction utility

---

# 🌐 FastAPI Backend

Main backend file:
- app.py

## Available Endpoints

| Endpoint        | Method | Description |
|---------------|--------|-------------|
| `/`           | GET    | Web UI |
| `/detect`     | POST   | Image detection (API) |
| `/detect-ui`  | POST   | Image detection (UI form) |
| `/detect-video` | POST | Video detection |

---

# 🛠️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

## 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
requirements.txt includes:

- torch==2.1.2
- torchvision==0.16.2
- numpy==1.26.4
- opencv-python==4.9.0.80
- pillow
- fastapi
- uvicorn
- python-multipart

▶️ Running the Application
Start FastAPI server:
```bash
uvicorn app:app --reload
```
🧠 Training the Models
Train Image Model
```bash
python train.py
```
Dataset structure:
```code
dataset/
 ├── real/
 └── fake/
```
Output:
```code
deepfake_model.pth
Train Video Model
python video_train.py
```
Dataset structure:
```code
video_dataset/
 ├── real/
 └── fake/
```
Output:
```code
video_deepfake_model.pth
```
⚙️ Tech Stack
Python
PyTorch
Torchvision
OpenCV
FastAPI
Uvicorn
Pillow
NumPy

📊 Features
Image Deepfake Detection
Video Deepfake Detection
REST API Support
Web UI
GPU Support (CUDA)
Pretrained Backbone
