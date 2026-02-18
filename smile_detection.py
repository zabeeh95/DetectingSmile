import numpy as np
import cv2

from tensorflow.keras import models

detector = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
model = models.load_model("data\model.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangle = detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)


    for (x, y, w, h) in face_rectangle:
        roi = frame_gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float32") / 255.0
        roi = np.array(roi)
        roi = roi.reshape(1, 28, 28, 1)
        # roi = np.expand_dims(roi, axis=0)       # (1, 28, 28)  adds batch
        # roi = np.expand_dims(roi, axis=-1)      # shape (1, 28, 28, 1)  adds channel

        notSmiling, smiling = model.predict(roi)[0]
        print("model output is ", notSmiling, smiling)
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Camera Window ", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
