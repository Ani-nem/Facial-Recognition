from typing import Optional
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship, sessionmaker
from pgvector.sqlalchemy import Vector
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
VECTOR_SIZE = 512

class Person(Base):
    #Table for Known People in dB
    __tablename__ = 'person'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column()
    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="person")

    def __repr__(self):
        return f"<Person(id={self.id}, name={self.name}, embeddings = {self.embeddings})>"

class Embedding(Base):
    #Stores an embedding alongside the confidence score with it
    __tablename__ = 'embedding'
    id: Mapped[int] = mapped_column(primary_key=True)
    embedding: Mapped[Vector] = mapped_column(Vector(VECTOR_SIZE))
    confidence : Mapped[float] = mapped_column()
    person_id: Mapped[int] = mapped_column(ForeignKey('person.id'))
    person: Mapped[Person] = relationship(back_populates="embeddings")

    def __repr__(self):
        return f"Embedding(id={self.id}, confidence = {self.confidence}, person_id={self.person_id}, person={self.person})"


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def get_db() -> Session:
    try :
        db = Session()
        yield Session()
    except Exception as e :
        print(f"Error: {str(e)}")
    finally:
        db.close()