from datetime import timedelta, datetime, timezone
from typing import Annotated
from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from flask.cli import load_dotenv
from pydantic import BaseModel, EmailStr, ConfigDict
import os
from .auth_config import bcrypt_context, user_dependency
from backend.database.db_config import db_dependency
from backend.database.models import User
from sqlalchemy.exc import IntegrityError
from jose import jwt, JWTError
from fastapi.params import Depends


load_dotenv()
SECRET_KEY = os.environ.get("AUTH_SECRET_KEY")
AUTH_ALGORITHM = os.environ.get("AUTH_ALGORITHM")
AUTH_ACCESS_EXPIRE_MINUTES = int(os.environ.get("AUTH_ACCESS_EXPIRE_MINUTES"))

router = APIRouter(prefix="/auth", tags=["auth"])

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str

class UserResponse(BaseModel): # Pydantic model for response
    id: int
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)

def authenticate_user(email: str, password: str, db : db_dependency) -> User | bool:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=AUTH_ACCESS_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=AUTH_ALGORITHM)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm,Depends()], db: db_dependency):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    access_token = create_access_token(data = {"sub": user.email, "id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

#TODO: consider moving to db.py instead
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: db_dependency): # Type hint db
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already exists")

        hashed_password = bcrypt_context.hash(user.password)
        new_user = User(email=user.email, password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    except IntegrityError as e:
        db.rollback()
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Email already exists")
        else:
            raise HTTPException(status_code=500, detail="Database error occurred")
    except HTTPException as e:  # Catch HTTPException
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred during registration:  {str(e)}") # Raise generic exception

@router.get("/me")
async def get_me(user: user_dependency):
    return user



