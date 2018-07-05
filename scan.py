from imutils.video import VideoStream
from imutils import resize
import face_recognition
import pickle
import cv2

data = pickle.loads(open("./encodings.dat", "rb").read())
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model="hog")
    faces = face_recognition.face_encodings(rgb, boxes)
    names = []

    for face in faces:
        matches = face_recognition.compare_faces(data["encodings"], face)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    for (top, right, bottom, left), name in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
