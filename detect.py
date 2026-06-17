import cv2
import numpy as np

net = cv2.dnn.readNet(
    "yolov3-tiny.weights",
    "yolov3-tiny.cfg"
)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

output_layers = []

for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])
    
camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame,
        0.00392,
        (416, 416),
        (0, 0, 0),
        True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.2:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])

                confidences.append(float(confidence))

                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.5,
        0.4
    )

    for i in range(len(boxes)):

        if i in indexes:

            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()