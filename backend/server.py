from fastapi import FastAPI

from backend.facialrecognition import FaceRecognitionModel
from backend.db import DataBaseModel

app = FastAPI()

desired_classes = ["person"]
database_model = DataBaseModel()
model = FaceRecognitionModel("yolo11n.pt", desired_classes, database_model)

@app.get("/")
async def root():
    return {"message": "Hello World"}
