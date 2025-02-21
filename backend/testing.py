from facialrecognition import FaceRecognitionModel
from database.db import DataBaseOps, DataBaseConnection
from database.db_config import get_db





db_conn = DataBaseConnection()

db_ops = DataBaseOps()
model = FaceRecognitionModel("yolo11n.pt", ["person"], db_ops)

with next(get_db()) as db:
    model.detect_people(db, "../datasets/testDataOnePerson")
    model.visualize_results(db)
