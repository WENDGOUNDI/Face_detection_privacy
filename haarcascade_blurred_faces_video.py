#Libraries importation
import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('video_test.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outpy2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:
    ret, frame = cap.read()
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_data = face_detect.detectMultiScale(frame, 1.3, 4)
    for (x, y, w, h) in face_data:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (25,25),cv2.BORDER_ISOLATED)
        # blur image
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        out.write(frame)
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()