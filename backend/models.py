from typing import Optional
from sqlalchemy import  ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from db_config import Base, VECTOR_SIZE


class Person(Base):
    #Table for Known People in dB
    __tablename__ = 'person'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column()
    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="person")

    def __repr__(self):
        return f"<Person(id={self.id}, name={self.name}, embeddings = {self.embeddings})>"

# TODO: remove confidence from Embedding & update functions
class Embedding(Base):
    #Stores an embedding alongside the confidence score with it
    __tablename__ = 'embedding'
    id: Mapped[int] = mapped_column(primary_key=True)
    embedding: Mapped[Vector] = mapped_column(Vector(VECTOR_SIZE))
    confidence : Mapped[float] = mapped_column()
    img_path: Mapped[Optional[str]] = mapped_column()
    person_id: Mapped[int] = mapped_column(ForeignKey('person.id'))
    person: Mapped[Person] = relationship(back_populates="embeddings")

    __table_args__ = (
        Index(
            'embedding_hnsw_index',
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )

    def __repr__(self):
        return f"Embedding(id={self.id}, confidence = {self.confidence}, person_id={self.person_id}, person={self.person})"
