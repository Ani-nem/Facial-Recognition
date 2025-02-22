from typing import Annotated, Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from facialrecognition import FaceRecognitionModel
from database.db import DataBaseOps
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from database.db_config import Base, engine, get_db

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

db_dependency = Annotated[Session, Depends(get_db)]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/people", response_model=List[PersonModel])
def get_people(db: db_dependency):
    people = database_model.get_people(db)
    return people
