from numpy import ndarray
from ultralytics import YOLO
from deepface import DeepFace
import cv2
from db import *

def get_class_ids(classes):
    """
    Returns a list of classes ids.

    :param classes: list of strings of classes to detect
    :return: list of ids of classes to detect
    """

    desired_ids_dict = {value: key for key, value in model.names.items()}
    desired_ids = list(map(lambda x: desired_ids_dict[x], classes))

    return desired_ids



def process_face(cropped_img : ndarray, db : Session):
    """
    Processes face into embedding and stores embedding into db.
    :param cropped_img: numpy array of face image
    :param db: db session
    :return: None
    """
    try:
        data = DeepFace.represent(img_path=cropped_img)
        embedding = data[0]["embedding"]
        confidence = data[0]["face_confidence"]

        similar_embedding, similar_person = similarity_search(db, embedding)

        if similar_embedding is not None:
            add_embedding(db, embedding, confidence, similar_person)
            print(f"Added embedding to: Person {similar_person.id}")
        else:
            new_person = register_person(db, embedding, confidence)
            print(f"Registered Person {new_person.id}")

    except Exception as e:
        print(f"Error occurred while generating embedding: {str(e)}")



def detect_people(directory : str):
    """
    Given a directory of images, detects people using Yolo and stores embeddings for each person in db
    :param directory: a directory of images to detect or path to a singular image
    :return:
    """

    with next(get_db()) as db:
        try:
            results = model.predict(source=directory, classes=desired_ids, save_crop=True,
                                project=SAVE_DATA_PATH, conf=0.8, max_det = 1, batch=8)

            for result in results:
                boxes = result.boxes
                img = result.orig_img


                #TODO: implement batch processing
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    process_face(cropped_img, db)

        except Exception as e:
            print(f"Error occurred while detecting person: {str(e)}")

def detect_people_video(video : str):
    """
    Detects people using Yolo and stores embeddings for each person in db
    :param video: path to video file
    :return: None
    """

    with next(get_db()) as db:
        try:
            results = model.predict(source=video, classes=desired_ids, stream=True, batch=8,
                                    save_crop=True, project=SAVE_DATA_PATH, conf=0.8, max_det = 1)
            for result in results:
                boxes = result.boxes
                img = result.orig_img
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    process_face(cropped_img, db)

        except Exception as e:
            print(f"Error occurred while detecting person: {str(e)}")

#Modifiable Constants
desired_classes = ["person"]
SOURCE_DATA_PATH = "./testData"
SAVE_DATA_PATH = "./Trials"


#Load the model and run inference on specified SOURCE directory
model = YOLO("yolo11n.pt")
desired_ids = get_class_ids(desired_classes)

# results = model.predict(source=SOURCE_DATA_PATH, classes = desired_ids, save_crop = True,
#                         project = SAVE_DATA_PATH, conf = 0.7)
#
# size = 0
# #For each result, obtain bounding box and path of saved image, and convert to embedding
# for result in results:
#     boxes = result.boxes
#     img = result.orig_img
#     size += len(boxes)
#
#
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         cropped_img = img[y1:y2, x1:x2]
#         try:
#             data = DeepFace.represent(img_path=cropped_img, model_name="Facenet512")
#             embedding = data[0]["embedding"]
#             confidence = data[0]["face_confidence"]
#
#             print(embedding, confidence)
#         except Exception as e:
#             print(f"{str(e)}")
#
# print(size)



#detect_people("./testData2/2019-disneylegend-rdj.jpg")

detect_people("./testDataOnePerson")
