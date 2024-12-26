from ultralytics import YOLO
from deepface import DeepFace


def get_class_ids(classes):
    """
    Returns a list of classes ids.

    :param classes: list of strings of classes to detect
    :return: list of ids of classes to detect
    """

    desired_ids_dict = {value: key for key, value in model.names.items()}
    desired_ids = list(map(lambda x: desired_ids_dict[x], classes))

    return desired_ids

#Modifiable Constants
desired_classes = ["person"]
SOURCE_DATA_PATH = "./testData"
SAVE_DATA_PATH = "./Trials"


#Load the model and run inference on specified SOURCE directory
model = YOLO("yolo11n.pt")
desired_ids = get_class_ids(desired_classes)
results = model.predict(source=SOURCE_DATA_PATH, classes = desired_ids, save_crop = True,
                        project = SAVE_DATA_PATH, conf = 0.7)

size = 0
#For each result, obtain bounding box and path of saved image, and convert to embedding
for result in results:
    boxes = result.boxes
    img = result.orig_img
    path = result.path
    size += len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_img = img[y1:y2, x1:x2]
        try:
            embedding = DeepFace.represent(img_path=cropped_img, model_name="Facenet512")
            print(embedding)
        except Exception as e:
            print(f"Could not find an embedding for file at {path}")

print(size)

