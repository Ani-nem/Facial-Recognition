from typing import Annotated
from jose import jwt, JWTError
from dotenv import load_dotenv
import os

from fastapi.params import Depends
from passlib.context import CryptContext
from fastapi import  HTTPException, status
from fastapi.security import OAuth2PasswordBearer

load_dotenv()
SECRET_KEY = os.environ.get("AUTH_SECRET_KEY")
AUTH_ALGORITHM = os.environ.get("AUTH_ALGORITHM")
AUTH_ACCESS_EXPIRE_MINUTES = int(os.environ.get("AUTH_ACCESS_EXPIRE_MINUTES"))

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer_dependency = Annotated[str, Depends(oauth2_bearer)]

async def get_current_user(token: oauth2_bearer_dependency):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[AUTH_ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("id")
        if email is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user credentials")
        return {"email": email, "id": user_id}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user credentials")

user_dependency = Annotated[dict, Depends(get_current_user)]