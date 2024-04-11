
from ultralytics import YOLO
import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Get the screen resolution
screen_width = 1920  # Set your screen width here
screen_height = 1080  # Set your screen height here

# Set the resolution for displaying the webcam feed
display_width = screen_width // 1
display_height = screen_height // 1

# Set the video writer for saving the output
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (display_width, display_height))

# Initialize the YOLO model
model = YOLO("../YOLO-Weights/yolov8x.pt")

# Class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    # Read frame from the webcam
    success, img = cap.read()

    # Resize the frame to fit the screen size
    img = cv2.resize(img, (display_width, display_height))

    # Perform object detection using YOLOv5
    results = model(img, stream=True)

    num = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                num += 1

                # Draw a rectangle around the detected person
                cv2.rectangle(img, (x1, y1), (x2, y2), (256, 0, 256), 2)
                # Display the count of each person
                cv2.putText(img, str(num), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(img)

    # Display the processed frame
    cv2.imshow("Image", img)

    # Check for key press and exit the loop if '1' is pressed
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Release the video capture and video writer objects
out.release()
cap.release()
cv2.destroyAllWindows()
