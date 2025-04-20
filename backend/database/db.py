from typing import Any, Generator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from .db_config import engine
from .models import Person, Embedding, User


# TODO: remove
class DataBaseConnection:
    def __init__(self):
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def get_db(self) -> Generator[Session, Any, None]:
        try:
            db = self.Session()
            yield db
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            db.close()


class DataBaseOps:
    @staticmethod
    def add_embedding(db: Session, embedding: list[float], confidence: float, img_path: str, person_id: int):
        """
        Adds embedding to the specified person
        :param db: database session
        :param img_path: path to image
        :param embedding: embedding vector
        :param confidence: confidence value
        :param person_id: id of the person
        :return: Updated person object, None if not found
        """
        try:
            # Add embedding to given person
            # noinspection PyTypeChecker
            db.add(Embedding(embedding=embedding, person_id=person_id, img_path=img_path, confidence=confidence))
            db.commit()

            person_statement = select(Person).where(Person.id == person_id)
            updated_person = db.execute(person_statement).scalars().first()
            return updated_person
        except Exception as e:
            print(f"Error adding embedding to person {person_id}: {str(e)}")
            db.rollback()

    @staticmethod
    def register_person(db: Session, embedding: list[float], img_path: str, confidence: float = 1.0):
        """
        Creates new unknown person with given embedding
        :param db: database session
        :param img_path: path to image
        :param embedding: embedding vector
        :param confidence: confidence value
        :return: Newly created Person Object
        """
        try:
            # Create new Person
            new_person = Person()
            db.add(new_person)
            db.flush()

            # Create new embedding and link to Person
            # noinspection PyTypeChecker
            new_embedding = Embedding(embedding=embedding, confidence=confidence, img_path=img_path, person=new_person)
            db.add(new_embedding)
            db.commit()

            person_statement = select(Person).where(Person.id == new_person.id)
            updated_person = db.execute(person_statement).scalars().first()
            return updated_person
        except Exception as e:
            print(f"Error registering new person: {str(e)}")
            db.rollback()

    @staticmethod
    def similarity_search(db: Session, orig_embedding: list[float], user_id: int):
        """
        Scans db for closest neighbour embedding of given vector using cosine distance
        :param db: database session
        :param orig_embedding: embedding vector
        :return: the closest embedding, the id of the person it belongs to, and the confidence of the match. None, None, None for no match.
        """
        try:

            #find valid people to look through for given user
            people_to_search = db.execute(select(Person.id).where(Person.user_id == user_id)).scalars().all()
            # Find similar embedding
            statement = (
                select(
                    Embedding,
                    Embedding.embedding.cosine_distance(orig_embedding)
                )
                .where(Embedding.embedding.cosine_distance(orig_embedding) <= 0.0658,
                       Embedding.person_id.in_(people_to_search))
                .order_by(Embedding.embedding.cosine_distance(orig_embedding))
                .limit(1))
            result = db.execute(statement).first()

            if result is None:
                return None, None, None
            else:
                closest_embedding_obj, distance = result
                similarity = 1 - distance
                closest_embedding = closest_embedding_obj.embedding
                closest_person_id = closest_embedding_obj.person.id
                return closest_embedding, closest_person_id, similarity

        except SQLAlchemyError as e:
            print(f"Database Error occurred during Similarity Search: {str(e)}")
            return None, None, None
        except Exception as e:
            print(f"An Error occurred during Similarity Search: {str(e)}")
            return None, None, None

    @staticmethod
    def get_people(db: Session, user_id: int):
        """
        Returns all people ids in the database connected to the current user
        :param db: database session, userid
        :return: list of people ids
        """
        try:
            statement = select(Person.id).where(Person.user_id == user_id)
            people = db.execute(statement).scalars().all()
            return people
        except Exception as e:
            print(f"Error retrieving people ids: {str(e)}")
            return []

    @staticmethod
    def create_person(db: Session, user_id: int):
        try:
            new_person = Person(user_id=user_id)
            db.add(new_person)
            db.flush()
            db.commit()
            return new_person.id
        except Exception as e:
            print(f"Error creating new person: {str(e)}")
            db.rollback()

    @staticmethod
    def get_person_images(db: Session, user_id: int, person: int):

        try:
            statement = select(Embedding.img_path).where(Embedding.person_id == person)
            result = db.execute(statement)

            paths = [row.img_path for row in result]
            return paths

        except Exception as e:
            print(f"Error retrieving image path: {str(e)}")
