from typing import Optional, List, Dict
import dotenv
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import mimetypes
from backend.facialrecognition import FaceRecognitionModel
from backend.database.db import DataBaseOps
from pydantic import BaseModel, ConfigDict
from backend.database.db_config import Base, engine, db_dependency
from backend.auth.auth import router as auth_router
from backend.auth.auth_config import user_dependency
import boto3
from botocore.client import Config
import os

app = FastAPI()
s3 = boto3.resource(
    's3',
    region_name="us-east-2")
s3_client = boto3.client(
    "s3",
    region_name="us-east-2",
    config=Config(signature_version='s3v4')
)
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


@app.get("/people", response_model=List[int])
def get_people(user: user_dependency, db: db_dependency):
    people = database_model.get_people(db, user.get("id"))
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
            content_type = mimetypes.guess_type(file_name)[0]
            if content_type is None:
                content_type = "application/octet-stream"
            if isinstance(image_data, bytes):
                image_rgb_format = decode_image_bgr(image_data)
                people = model.detect_people(db, image_rgb_format)
                for person in people:

                    rgb_img = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if encodings:
                        embedding = encodings[0]
                        closest_embedding, person_id, similarity = database_model.similarity_search(db, embedding,
                                                                                                    user_id)

                        # TODO specify the type of the thing we are sending
                        if closest_embedding is None:
                            person_id = database_model.create_person(db, user_id)
                            key = f"{user_id}/{person_id}/{file_name}"
                            bucket.put_object(Key=key, Body=image_data, ContentType=content_type)
                            database_model.add_embedding(db, embedding, 1.0, key, person_id)
                        else:
                            key = f"{user_id}/{person_id}/{file_name}"
                            bucket.put_object(Key=key, Body=image_data, ContentType=content_type)
                            database_model.add_embedding(db, embedding, similarity, key, person_id)

    except Exception as e:
        print(f"Error in uploading image(s): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# returns a dictionary with key = person id, value = list of image paths
@app.get("/api/all_keys", response_model=Dict[int, List[str]])
def get_all_keys(user: user_dependency, db: db_dependency):
    user_id = user.get("id")

    try:
        people = get_people(user, db)
        imageDict = {}

        for person_id in people:
            paths = database_model.get_person_images(db, user_id, person_id)
            imageDict[person_id] = paths

        return imageDict

    except Exception as e:
        print(f"Error in getting all keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/images/view/{key:path}")
async def get_image(user: user_dependency, key: str):
    try:
        obj = bucket.Object(key)
        response = obj.get()
        image_data = response["Body"].read()
        image_type = response["ContentType"]
        return Response(content=image_data, media_type=image_type)

    except Exception as e:
        print(f"Error in retrieving image for key: {key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/images/download/{key:path}")
async def download_image(user: user_dependency, key: str):
    try:
        obj = bucket.Object(key)
        response = obj.get()
        image_data = response["Body"].read()
        image_type = response["ContentType"]
        filename = key.split("/")[-1]
        return Response(
            content=image_data,
            media_type=image_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        print(f"Error in downloading image for key: {key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/grouped")
def get_grouped_image_urls(user: user_dependency, db: db_dependency):
    try:
        user_id = user.get("id")
        people = database_model.get_people(db, user_id)
        image_dict = {}

        for person_id in people:
            keys = database_model.get_person_images(db, user_id, person_id)

            signed_urls = []
            for key in keys:
                url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                    ExpiresIn=600
                )
                signed_urls.append({
                    "url": url,
                    "filename": key.split("/")[-1],
                    "key": key
                })

            image_dict[person_id] = signed_urls

        return image_dict

    except Exception as e:
        print(f"Error grouping images: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve grouped images.")
