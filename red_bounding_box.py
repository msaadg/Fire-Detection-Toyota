import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO('trained-models/best-luminous.pt')  # Update the path if necessary

    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read a frame from the camera.")
            break

        # Resize frame for consistency
        frame = cv2.resize(frame, (980, 800))

        # Process the frame with YOLO
        results = model.predict(source=frame, conf=0.25, save=False, show=False)

                 # Extract bounding boxes from YOLO results
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert boxes to numpy array for faster processing
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()


          # Convert the frame to a copy so we can modify it without affecting the original
        frame_with_boxes = frame.copy()

                #   Vectorized approach: draw bounding boxes in bulk
        for box, class_id in zip(boxes.astype(int), classes.astype(int)):  # Convert to integer coordinates
         # Model class names: {0: 'fire', 1: 'smoke'}
            color = (255,0, 0)  # default: blue 
            if model.names[class_id] == 'fire':
                color = (0, 0, 255) # Gray for smoke
            # else:
            #     color = (0, 0, 255)  # Red for other detections (e.g., fire)

            # Draw bounding box with the selected color
            cv2.rectangle(frame_with_boxes, (box[0], box[1]), (box[2], box[3]), color, 2)  # Color as per the object class

        # Add class labels and confidence to the bounding boxes
        for box, class_id, confidence in zip(boxes, classes, confidences):
            # Ensure box coordinates are integers
            x, y = int(box[0]), int(box[1])

            label = f'{model.names[int(class_id)]} {confidence:.2f}'
            # Put the label above the bounding box
            cv2.putText(frame_with_boxes, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)





        # Convert YOLO results to OpenCV format
        # frame_with_boxes = results[0].plot()

        #   # Resize processed frame to match the original frame size
        processed_frame = cv2.resize(frame_with_boxes, (980, 800))

        # Concatenate the original and processed frames side by side
        combined_frame = np.hstack((frame, processed_frame))


        # Display the resulting frame
        cv2.imshow('Unprocessed (Left) | Processed (Right)', combined_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
