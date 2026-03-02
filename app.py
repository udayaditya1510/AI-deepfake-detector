from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import shutil
import os
from predict import predict_image

app = FastAPI(title="AI Image & Deepfake Detector")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = predict_image(file_path)
    return {"result": label, "confidence": confidence}

@app.post("/detect-ui", response_class=HTMLResponse)
async def detect_ui(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = predict_image(file_path)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": label,
            "confidence": f"{confidence}%"
        }
    )

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = predict_video(path)
    return {"result": result}
