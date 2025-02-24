from typing import  Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.facialrecognition import FaceRecognitionModel
from backend.database.db import DataBaseOps
from pydantic import BaseModel, ConfigDict
from backend.database.db_config import Base, engine, db_dependency
from backend.auth.auth import router as auth_router
from backend.auth.auth_config import user_dependency



app = FastAPI()
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


app.include_router(auth_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/people", response_model=List[PersonModel])
def get_people(db: db_dependency):
    people = database_model.get_people(db)
    return people


@app.get("/hello")
def hello(user: user_dependency):
    return {"email": user.get("email"), "id": user.get("id")}