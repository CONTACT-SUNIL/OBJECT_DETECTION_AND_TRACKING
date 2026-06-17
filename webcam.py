import cv2

camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()