import cv2

def extract_faces(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    faces = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            faces.append(face)
            count += 1
            break  # one face per frame

    cap.release()
    return faces
