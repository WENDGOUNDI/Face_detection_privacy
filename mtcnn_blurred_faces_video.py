from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video_test.mp4')
detector = MTCNN()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)
    # get the context for drawing boxes
    # plot each box
    for result in faces:
        # Extract coordinates from faces
        values_view = result.values()
        value_iterator = iter(values_view)
        first_value = next(value_iterator)
        # Set variables as x,y,w,h format
        x,y,w,h = first_value
        #print(x,y,w,h)
        # Apply the blur
        roi = frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (25,25),cv2.BORDER_ISOLATED)
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('Face_detected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()