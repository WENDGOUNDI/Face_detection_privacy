from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2

# load the image
img = cv2.imread('face_3.png')
detector = MTCNN()
faces = detector.detect_faces(img)
# get the context for drawing boxes
#ax = pyplot.gca()
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
    roi = img[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (25,25),cv2.BORDER_ISOLATED)
    img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Face_detected', img)
cv2.imwrite("blured_faces.jpg",img)
cv2.waitKey()
cv2.destroyAllWindows()