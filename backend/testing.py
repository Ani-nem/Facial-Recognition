from facialrecognition import FaceRecognitionModel
from db import DataBaseOps, DataBaseConnection




db_conn = DataBaseConnection()

db_ops = DataBaseOps()
model = FaceRecognitionModel("yolo11n.pt", ["person"], db_ops)

with next(db_conn.get_db()) as db:
    model.detect_people(db, "../datasets/testDataOnePerson")
    model.visualize_results(db)
