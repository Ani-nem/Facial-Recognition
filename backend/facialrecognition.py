from numpy import ndarray
from ultralytics import YOLO
import cv2
from db import DataBaseOps, Session
from util import *
import os


class FaceRecognitionModel:
    SOURCE_DATA_PATH = "../datasets/testData"
    SAVE_DATA_PATH = "../datasets/Trials"

    def __init__(self, model_path: str, classes: list[str] , db_ops: DataBaseOps):
        self.model = YOLO(model_path)
        desired_ids_dict = {value: key for key, value in self.model.names.items()}
        self.desired_ids = list(map(lambda x: desired_ids_dict[x], classes))
        self.db_ops = db_ops

    def get_class_ids(self, classes: list[str]):
        """
        Returns a list of classes ids.

        :param classes: list of strings of classes to detect
        :return: list of ids of classes to detect
        """

        desired_ids_dict = {value: key for key, value in self.model.names.items()}
        desired_ids = list(map(lambda x: desired_ids_dict[x], classes))

        return desired_ids


    # Dev helper, not needed for production
    #TODO: Remove later in production
    def visualize_results(self, db: Session, parent_directory: str = "../datasets/TrialsOrganized"):

        people = self.db_ops.get_people(db)
        for person in people:
            person_dir = os.path.join(parent_directory, f"person_{person.id}")
            os.makedirs(person_dir, exist_ok=True)
            embeddings = person.embeddings
            for embedding in embeddings:
                img_path = embedding.img_path
                img = cv2.imread(img_path)
                if img is not None:
                    img_name = os.path.basename(img_path)
                    new_img_path = os.path.join(person_dir, img_name)
                    cv2.imwrite(new_img_path, img)

    def process_face(self, db: Session, cropped_img: ndarray, img_path: str = None):
        """
        Processes face into embedding and stores embedding into db.
        :param cropped_img: numpy array of face image, or relative path to image
        :param db: db session
        :param img_path: path to image
        :return: None
        """
        try:
            # Convert the image from BGR to RGB (face_recognition expects RGB)
            rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            # Get face encoding using face_recognition instead of DeepFace
            face_encodings = face_recognition.face_encodings(rgb_img)

            if len(face_encodings) == 0:
                print("No face detected in the image")
                return

            # Get the first face encoding (assuming one face per image)
            embedding = face_encodings[0].tolist()


            similar_embedding, similar_person, confidence = self.db_ops.similarity_search(db, embedding)

            if similar_embedding is not None:
                self.db_ops.add_embedding(db, embedding, confidence, img_path, similar_person)
                print(f"Added embedding to: Person {similar_person.id}")
            else:
                new_person = self.db_ops.register_person(db, embedding, img_path)
                print(f"Registered Person {new_person.id}")

        except Exception as e:
            print(f"Error occurred while generating embedding: {str(e)}")

    #TODO: Add functionality to detect boxes and animals too
    def detect_people(self, db:Session, directory: str):
        """
        Given a directory of images, detects people using Yolo and stores embeddings for each person in db
        :param db: database session
        :param directory: a directory of images to detect or path to a singular image
        :return:
        """

        try:
            results = self.model.predict(source=directory, classes=self.desired_ids, save_crop=True,
                                         project=self.SAVE_DATA_PATH, conf=0.8, max_det=1, batch=8)

            for result in results:
                boxes = result.boxes
                img = result.orig_img
                orig_img_path = result.path

                # TODO: implement batch processing
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    crop_img_path = (f"{self.SAVE_DATA_PATH}/predict/crops/person/"
                                     f"{os.path.splitext(os.path.basename(orig_img_path))[0]}.jpg")
                    print(crop_img_path)
                    self.process_face(db, cropped_img, crop_img_path)

        except Exception as e:
            print(f"Error occurred while detecting person: {str(e)}")


    def detect_people_video(self, db: Session, video: str):
        """
        Detects people using Yolo and stores embeddings for each person in db
        :param db: database session
        :param video: path to video file
        :return: None
        """

        try:
            results = self.model.predict(source=video, classes=self.desired_ids, stream=True, batch=8,
                                         save_crop=True, project=self.SAVE_DATA_PATH, conf=0.8, max_det=1)
            for result in results:
                boxes = result.boxes
                img = result.orig_img
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    self.process_face(db, cropped_img)

        except Exception as e:
            print(f"Error occurred while detecting person: {str(e)}")


# dataset_path = "lfw_filtered"
# stats = calculate_similarity_statistics(dataset_path)
# print_similarity_stats(stats)

# copy_folders_with_images("./lfw_funneled", "./lfw_filtered", 10)

# Modifiable Constants
# desired_classes = ["person"]
# # Load the model and run inference on specified SOURCE directory
# database_model = DataBaseOps()
# model = FaceRecognitionModel("yolo11n.pt", desired_classes, database_model)
# model.detect_people("../datasets/testDataOnePerson")
# model.visualize_results()
