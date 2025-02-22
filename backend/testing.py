from facialrecognition import FaceRecognitionModel
from database.db import DataBaseOps, DataBaseConnection
from database.db_config import get_db, Base, engine




Base.metadata.create_all(engine)
db_conn = DataBaseConnection()

db_ops = DataBaseOps()
model = FaceRecognitionModel("yolo11n.pt", ["person"], db_ops)

with next(get_db()) as db:
    model.detect_people(db, "../datasets/testData")
    model.visualize_results(db)
