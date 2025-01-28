from sys import orig_argv

from numpy import ndarray
from ultralytics import YOLO
from deepface import DeepFace
import cv2
from db import *


import os
import numpy as np
from deepface import DeepFace
from collections import defaultdict


def get_class_ids(classes):
    """
    Returns a list of classes ids.

    :param classes: list of strings of classes to detect
    :return: list of ids of classes to detect
    """

    desired_ids_dict = {value: key for key, value in model.names.items()}
    desired_ids = list(map(lambda x: desired_ids_dict[x], classes))

    return desired_ids

#Dev helper, not needed for production
def visualize_results(parent_directory: str = "./TrialsOrganized"):
    with next(get_db()) as db:
        people = get_people(db)
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

def process_face(cropped_img : ndarray, db : Session, img_path : str = None):
    """
    Processes face into embedding and stores embedding into db.
    :param cropped_img: numpy array of face image, or relative path to image
    :param db: db session
    :param img_path: path to image
    :return: None
    """
    try:
        data = DeepFace.represent(img_path=cropped_img, model_name="VGG-Face")
        embedding = data[0]["embedding"]
        confidence = data[0]["face_confidence"]

        similar_embedding, similar_person = similarity_search(db, embedding)

        if similar_embedding is not None:
            add_embedding(db, embedding, confidence, img_path, similar_person)
            print(f"Added embedding to: Person {similar_person.id}")
        else:
            new_person = register_person(db, embedding, confidence, img_path)
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
                orig_img_path = result.path


                #TODO: implement batch processing
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    crop_img_path = (f"{SAVE_DATA_PATH}/predict/crops/person/"
                                     f"{os.path.splitext(os.path.basename(orig_img_path))[0]}.jpg")
                    print(crop_img_path)
                    process_face(cropped_img, db, crop_img_path)

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

def facial_detection(directory: str):
    """
    uses onlt the facial recognition model to detect faces in images
    :param directory: path to the directory containing images
    :return: None
    """
    with next(get_db()) as db:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    full_path = os.path.join(directory, file)
                    process_face(full_path,db, full_path)





def calculate_similarity_statistics(dataset_path: str):
    """
    Analyzes a dataset where each subfolder contains images of the same person.
    Calculates similarity statistics between images known to be of the same person.

    Dataset structure:
    dataset_path/
        person1/
            image1.jpg
            image2.jpg
        person2/
            image1.jpg
            image2.jpg

    :param dataset_path: Path to the root directory containing person folders
    :return: Dictionary containing similarity statistics
    """
    # Dictionary to store embeddings for each person
    person_embeddings = defaultdict(list)

    # Dictionary to store similarity scores for each person
    person_similarities = defaultdict(list)

    # Process each person's folder
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        print(f"\nProcessing person: {person_folder}")

        # Get embeddings for all images of this person
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(person_path, img_file)
            try:
                # Extract embedding
                data = DeepFace.represent(img_path=img_path, model_name="VGG-Face")
                embedding = data[0]["embedding"]
                person_embeddings[person_folder].append(embedding)
                print(f"Processed {img_file}")

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

    # Calculate similarities between all pairs of embeddings for each person
    overall_similarities = []

    for person, embeddings in person_embeddings.items():
        print(f"\nCalculating similarities for {person}")

        # Compare each embedding with every other embedding of the same person
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Calculate cosine similarity
                similarity = 1 - (np.dot(embeddings[i], embeddings[j]) /
                                  (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
                person_similarities[person].append(similarity)
                overall_similarities.append(similarity)

    # Calculate statistics
    stats = {
        'per_person': {},
        'overall': {}
    }

    # Per-person statistics
    for person, similarities in person_similarities.items():
        if similarities:  # Check if we have any similarities for this person
            stats['per_person'][person] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'count': len(similarities)
            }

    # Overall statistics
    if overall_similarities:
        stats['overall'] = {
            'mean': np.mean(overall_similarities),
            'std': np.std(overall_similarities),
            'min': np.min(overall_similarities),
            'max': np.max(overall_similarities),
            'count': len(overall_similarities)
        }

        # Calculate potential threshold suggestions
        stats['suggested_thresholds'] = {
            'strict': stats['overall']['mean'] - stats['overall']['std'],  # More strict threshold
            'balanced': stats['overall']['mean'],  # Balanced threshold
            'lenient': stats['overall']['mean'] + stats['overall']['std']  # More lenient threshold
        }

    return stats


def print_similarity_stats(stats):
    """
    Prints the similarity statistics in a readable format
    """
    print("\n=== Overall Statistics ===")
    print(f"Number of comparisons: {stats['overall']['count']}")
    print(f"Mean similarity: {stats['overall']['mean']:.4f}")
    print(f"Standard deviation: {stats['overall']['std']:.4f}")
    print(f"Range: {stats['overall']['min']:.4f} to {stats['overall']['max']:.4f}")

    print("\n=== Suggested Thresholds ===")
    print(f"Strict (fewer false positives): {stats['suggested_thresholds']['strict']:.4f}")
    print(f"Balanced: {stats['suggested_thresholds']['balanced']:.4f}")
    print(f"Lenient (fewer false negatives): {stats['suggested_thresholds']['lenient']:.4f}")

    print("\n=== Per-Person Statistics ===")
    for person, person_stats in stats['per_person'].items():
        print(f"\n{person}:")
        print(f"  Comparisons: {person_stats['count']}")
        print(f"  Mean: {person_stats['mean']:.4f}")
        print(f"  Std: {person_stats['std']:.4f}")
        print(f"  Range: {person_stats['min']:.4f} to {person_stats['max']:.4f}")


# Usage example:

# dataset_path = "lfw_funneled"
# stats = calculate_similarity_statistics(dataset_path)
# print_similarity_stats(stats)


detect_people("./testData2")
#
# #facial_detection("./testData")
visualize_results()
