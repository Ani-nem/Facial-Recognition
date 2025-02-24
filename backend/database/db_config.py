from typing import Generator, Any, Annotated

from fastapi.params import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env
DATABASE_USER = os.environ.get("DATABASE_USER")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")
DATABASE_HOST = os.environ.get("DATABASE_HOST")
DATABASE_PORT = os.environ.get("DATABASE_PORT")
DATABASE_NAME = os.environ.get("DATABASE_NAME")

if not all([DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT, DATABASE_NAME]):
    raise ValueError("Missing database environment variables")


DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
VECTOR_SIZE = 128

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, Any, None]:
    try:
        db = Session()
        yield db
    except Exception as e:
        print(f"Error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]