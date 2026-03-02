#🛡️ AI Image & Deepfake Detector
An end-to-end Deepfake Detection System built using PyTorch and FastAPI that can detect:
🖼️ AI-generated / manipulated Images
🎥 AI-generated / manipulated Videos
This project includes:
Custom-trained CNN model for image detection
CNN + LSTM model for video detection
FastAPI backend
Web UI support
REST API endpoints

🚀 Project Architecture:
User → FastAPI Server → Model Inference → Prediction Result
Image detection → ResNet18 classifier
Video detection → ResNet18 (feature extractor) + LSTM

🖼️ Image Deepfake Detection:
📌 Model Architecture:
Image detection uses a fine-tuned ResNet18 model:
Pretrained on ImageNet
Final layer modified for 2-class classification
Output:
REAL
FAKE (AI Generated)
Model definition:
model.py
Prediction script:
predict.py
Training script:
train.py

🌐 FastAPI Backend
Backend implementation:
app.py
Available Endpoints
| Endpoint       | Method         |  Description                  |
|----------------|----------------|-------------------------------|
| /              | GET            |  Web UI                       |
| /detect        | POST           |  Image detection (API)        |
| /detect-ui     | POST           |  Image detection (UI form)    |
| /detect-video  | POST           |  Video detection              |


🛠️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
Requirements file:
requirements
pip install -r requirements.txt
▶️ Running the Application
Start FastAPI Server
uvicorn app:app --reload
Open in browser:
http://127.0.0.1:8000

🧠 Training the Models
Train Image Model
python train.py
Dataset structure:
dataset/
 ├── real/
 └── fake/
Saves:
deepfake_model.pth
Train Video Model
python video_train.py
Dataset structure:
video_dataset/
 ├── real/
 └── fake/
Saves:
video_deepfake_model.pth

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
✅ Image Deepfake Detection
✅ Video Deepfake Detection
✅ REST API Support
✅ Web UI
✅ GPU Support (CUDA)
✅ Pretrained Backbone

📌 Future Improvements
Face alignment before prediction
Multiple face support per frame
Transformer-based video model
Deployment with Docker
Streamlit/React frontend

👨‍💻 Author
Uday Aditya
If you like this project ⭐ star the repository!
