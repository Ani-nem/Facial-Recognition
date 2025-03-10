from typing import Optional
from sqlalchemy import create_engine, ForeignKey, select, Index
from sqlalchemy.exc import SQLAlchemyError
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
VECTOR_SIZE = 4096

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
    img_path: Mapped[str] = mapped_column()
    person_id: Mapped[int] = mapped_column(ForeignKey('person.id'))
    person: Mapped[Person] = relationship(back_populates="embeddings")

    # __table_args__ = (
    #     Index(
    #         "embedding_embedding_idx",
    #         embedding,
    #         postgresql_using="hnsw",
    #         postgresql_ops={"embedding": "vector_cosine_ops"}
    #     ),
    # )

    def __repr__(self):
        return f"Embedding(id={self.id}, confidence = {self.confidence}, person_id={self.person_id}, person={self.person})"


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def get_db() -> Session:
    try :
        db = Session()
        yield db
    except Exception as e :
        print(f"Error: {str(e)}")
    finally:
        db.close()

def add_embedding(db: Session, embedding: list[float], confidence: float, img_path: str, person : Person):
    """
    Adds embedding to the specified person
    :param db: database session
    :param img_path: path to image
    :param embedding: embedding vector
    :param confidence: confidence value
    :param person: person
    :return: Updated person object, None if not found
    """
    try:
        #Add embedding to given person
        # noinspection PyTypeChecker
        db.add(Embedding(embedding=embedding, person=person, img_path=img_path, confidence=confidence))
        db.commit()

        person_statement = select(Person).where(Person.id == person.id)
        updated_person = db.execute(person_statement).scalars().first()
        return updated_person
    except Exception as e :
        print(f"Error adding embedding to {person.name}: {str(e)}")
        db.rollback()

def register_person(db: Session, embedding: list[float],  confidence: float, img_path : str):
    """
    Creates new unknown person with given embedding
    :param db: database session
    :param img_path: path to image
    :param embedding: embedding vector
    :param confidence: confidence value
    :return: Newly created Person Object
    """
    try:
        #Create new Person
        new_person = Person()
        db.add(new_person)
        db.flush()

        #Create new embedding and link to Person
        # noinspection PyTypeChecker
        new_embedding = Embedding(embedding=embedding, confidence=confidence, img_path=img_path, person = new_person)
        db.add(new_embedding)
        db.commit()

        person_statement = select(Person).where(Person.id == new_person.id)
        updated_person = db.execute(person_statement).scalars().first()
        return updated_person
    except Exception as e :
        print(f"Error registering new person: {str(e)}")

def similarity_search(db: Session, orig_embedding : list[float]):
    """
    Scans db for closest neighbour embedding of given vector using cosine distance
    :param db: database session
    :param orig_embedding: embedding vector
    :return: the closest embedding, and the person it belongs to, or None for no match.
    """
    try:

        #Find similar embedding
        statement = (
            select(Embedding)
            .where(1 - Embedding.embedding.cosine_distance(orig_embedding) >= 0.58)
            .order_by(1- Embedding.embedding.cosine_distance(orig_embedding))
            .limit(1))
        closest_embedding_obj = db.execute(statement).scalars().first()

        if closest_embedding_obj is None:
            return None, None
        else:
            closest_embedding = closest_embedding_obj.embedding
            closest_person = closest_embedding_obj.person
            return closest_embedding, closest_person

    except SQLAlchemyError as e :
        print(f"Database Error occurred during Similarity Search: {str(e)}")
        return None, None
    except Exception as e :
        print(f"An Error occurred during Similarity Search: {str(e)}")
        return None, None

def get_people(db: Session):
    """
    Returns all people in the database
    :param db: database session
    :return: list of people
    """
    try:
        statement = select(Person)
        people = db.execute(statement).scalars().all()
        return people
    except Exception as e:
        print(f"Error getting people: {str(e)}")
        return []
