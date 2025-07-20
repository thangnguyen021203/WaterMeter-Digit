from fastapi import FastAPI, File, UploadFile
from typing import List
from pipeline import DigitRecognizer

# Initialize the recognizer with your model paths
recognizer = DigitRecognizer(
    obb_model_path='model/Stage1-BoundingBox/model-1.pt',         
    orientation_model_path='model/Stage2-Orientation/model-2.pth', 
    digit_model_path='model/Stage3-Digit/model-3.pt', 
    orientation_classes=['bot', 'left', 'right', 'top']
)

app = FastAPI()

@app.post("/predict")
async def predict(images: List[UploadFile] = File(...)):
    results = []
    for image in images:
        image_bytes = await image.read()
        sequence = recognizer.process_image(image_bytes, image.filename)
        results.append({"filename": image.filename, "digits": sequence})
    return {"results": results} 