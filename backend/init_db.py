from db import Base, engine

def initialize_db():
    Base.metadata.create_all(engine)
    print("Database initialized")

if __name__ == "__main__":
    initialize_db()