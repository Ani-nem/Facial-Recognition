from typing import  Optional, List
import dotenv
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.facialrecognition import FaceRecognitionModel
from backend.database.db import DataBaseOps
from pydantic import BaseModel, ConfigDict
from backend.database.db_config import Base, engine, db_dependency
from backend.auth.auth import router as auth_router
from backend.auth.auth_config import user_dependency
import boto3
import os



app = FastAPI()
s3 = boto3.resource('s3')
dotenv.load_dotenv()
bucket_name = os.environ['AWS_S3_BUCKET_NAME']
bucket = s3.Bucket(bucket_name)


origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PersonModel(BaseModel):
    id: int
    name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


Base.metadata.create_all(engine)
desired_classes = ["person"]
database_model = DataBaseOps()
model = FaceRecognitionModel("yolo11n.pt", desired_classes, database_model)

def decode_image_bgr(image_data: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_data, np.uint8)
    image_bgr_format = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_bgr_format

def decode_image_rgb(image_data: bytes) -> np.ndarray:
    image_bgr_format = decode_image_bgr(image_data)
    image_rgb_format = cv2.cvtColor(image_bgr_format, cv2.COLOR_BGR2RGB)
    return image_rgb_format

app.include_router(auth_router)

@app.get("api/")
async def root():
    return {"message": "Hello World"}


@app.get("/people", response_model=List[PersonModel])
def get_people(db: db_dependency):
    people = database_model.get_people(db)
    return people


@app.get("/hello")
def hello(user: user_dependency):
    return {"email": user.get("email"), "id": user.get("id")}

@app.post("/api/upload_images")
async def upload_images(user: user_dependency, db: db_dependency, files: List[UploadFile] = File(...)):
    user_id = user.get("id")

    try:
        # for file in files:
        #     image_data = await file.read()
        #     image_rgb_format = decode_image_bgr(image_data)
        #     cv2.imshow('Uploaded Image', image_rgb_format)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     people = model.detect_people(db, image_rgb_format)

        for file in files:
            image_data = await file.read()
            file_name = file.filename
            if isinstance(image_data, bytes):
                image_rgb_format = decode_image_bgr(image_data)
                people = model.detect_people(db, image_rgb_format)
                for person in people:

                    rgb_img = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if encodings:
                        embedding = encodings[0]
                        closest_embedding, person_id, similarity = database_model.similarity_search(db, embedding, user_id)

                        if closest_embedding is None:
                            person_id = database_model.create_person(db, user_id)
                            key = f"{user_id}/{person_id}/{file_name}"
                            # bucket.put_object(Key=key, Body=image_data)
                            database_model.add_embedding(db, embedding, 1.0, key, person_id)
                        else:
                            key = f"{user_id}/{person_id}/{file_name}"
                            # bucket.put_object(Key=key, Body=image_data)
                            database_model.add_embedding(db, embedding, similarity, key, person_id)

    except Exception as e:
        print(f"Error in uploading image(s): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))