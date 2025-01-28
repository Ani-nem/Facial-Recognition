import os
import shutil
from collections import defaultdict

import face_recognition
import numpy as np



def copy_folders_with_images(parent_folder, new_folder, min_images):
    """
    Copies subfolders from the parent folder to the new folder if they contain more than min_images.

    Args:
        parent_folder (str): Path to the parent folder containing subfolders.
        new_folder (str): Path to the new folder (should not exist beforehand).
        min_images (int): Minimum number of images a folder must have to be copied.

    Returns:
        None
    """
    # Check if the parent folder exists
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"Parent folder does not exist: {parent_folder}")

    # Create the new folder if it does not exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Iterate through all subfolders in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Count the number of image files in the subfolder
            image_files = [
                f for f in os.listdir(subfolder_path)
                if os.path.isfile(os.path.join(subfolder_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
            ]

            # Copy the folder if it contains more than min_images
            if len(image_files) > min_images:
                destination_path = os.path.join(new_folder, subfolder_name)
                shutil.copytree(subfolder_path, destination_path)

    print(f"Folders with more than {min_images} images have been copied to: {new_folder}")

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
                # Load the image
                image = face_recognition.load_image_file(img_path)
                # Extract embedding
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    embedding = encodings[0]
                    person_embeddings[person_folder].append(embedding)
                    print(f"Processed {img_file}")
                else:
                    print(f"No face found in {img_file}")

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
                similarity = (np.dot(embeddings[i], embeddings[j]) /
                                  (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
                person_similarities[person].append(similarity)
                overall_similarities.append(similarity)
                print(f"Similarity between {i} and {j} for {person}: {similarity}")

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