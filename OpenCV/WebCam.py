import cv2

cap = cv2.VideoCapture(0)
# If you want to play video
# cap = cv2.VideoCapture('path.mp4')

while True:
    ret, image = cap.read()
    if not ret:
        break

    cv2.imshow('Title', image)
    cv2.waitKey(0)