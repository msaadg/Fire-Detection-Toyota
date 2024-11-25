import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO('trained-models/best-luminous.pt')  # Update the path if necessary

    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(1)

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
        frame = cv2.resize(frame, (1280, 960))

        # Process the frame with YOLO
        results = model.predict(source=frame, conf=0.25, save=False, show=False)

        # Convert YOLO results to OpenCV format
        processed_frame = results[0].plot()

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
