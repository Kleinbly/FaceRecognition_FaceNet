import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_facenet import FaceNet


def extractFaces(file_path, output_image_size=(160, 160)):
    # Load the HaarCascade model XLM file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    img = cv2.imread(file_path)
    # Convert the image into a numpy array
    img = np.asarray(img)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(img, scaleFactor=float(1.2), minNeighbors=10)

    # Loop through all the faces detected by the HaarCascade Model
    for (x, y, w, h) in faces:
        # Extract the region of the image corresponding to the detected face
        detected_face = img[y:y + h, x:x + w]

        try:
            # Resize the image according to the requirements
            detected_face = cv2.resize(detected_face, dsize=output_image_size)

        except cv2.error as e:
            raise ValueError(f"Error resizing image: {e}")

        return detected_face

    return None


def load_faces(folder_path):
    # Initialize an empty list to store face images
    faces = {}
    counter = 0
    # Iterate over all files in the given folder
    for file in os.listdir(folder_path):

        # Construct the full path to the file
        path = folder_path + file

        # Check if the current item is a file
        if os.path.isfile(path):
            # Extract faces from the image at the given path
            face_img = extractFaces(path)

            # If face extraction was successful (i.e., a face was found), add it to the faces dictionary
            if face_img is not None:
                # file corresponds the name of the face owner
                faces[file] = face_img
                counter += 1

        if counter >= 40:
            # We only need 40 images per person
            break

    # Return the dictionary celebrities and their respective faces
    return faces


def load_dataset():
    # Parent directory path
    parent_directory = 'Faces-Dataset/'

    # Initialize empty lists to store face images (X) and corresponding labels (Y)
    # X and Y store objects of type numpy.array
    X = []
    Y = []

    # Iterate over each sub-directory in the given folder
    for sub_dir in os.listdir(parent_directory):
        # Construct the full path to the sub-directory
        path = parent_directory + sub_dir + '/'

        # Load faces from the sub-directory
        face_dict = load_faces(path)

        # Load the face images
        detected_faces = face_dict.values()

        # Load the labels for the detected faces
        labels = face_dict.keys()

        # Extend the face images list (X) with the detected faces
        X.extend(detected_faces)

        # Extend the labels list (Y) with the created labels
        Y.extend(labels)

    # Convert the lists to numpy arrays for efficient numerical operations
    X_final = np.asarray(X)
    Y_final = np.asarray(Y)

    # Split the dataset into Train & Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.25, random_state=17)

    # Save the arrays to an NPZ file
    np.savez('generated_images.npz', array1=X_train, array2=X_test, array3=Y_train, array4=Y_test)

if __name__ == "__main__":
    load_dataset()


