from imutils import paths
import face_recognition
import pickle
import cv2
import os

data = {"encodings": [], "names": []}
imageset_path = list(paths.list_images("./imagesets"))

for source_image in imageset_path:
    name = source_image.split(os.path.sep)[-2]
    print("[INFO] processing image {}".format(source_image))
    image = cv2.imread(source_image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding in encodings:
        data['encodings'].append(encoding)
        data['names'].append(name)

f = open("encodings.dat", "wb")
f.write(pickle.dumps(data))
f.close()
