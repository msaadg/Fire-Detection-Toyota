import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

class FireDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detection Application")

        # Initialize YOLO model
        self.model = YOLO('trained-models/best-luminous.pt')  # Update the path if necessary

        # Initialize camera
        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Set up UI components
        self.init_ui()

        # Set up timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        # Labels to display the images
        self.label_unprocessed = QLabel(self)
        self.label_processed = QLabel(self)

        # Buttons
        self.btn_start = QPushButton('Start', self)
        self.btn_stop = QPushButton('Stop', self)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # Layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label_unprocessed)
        h_layout.addWidget(self.label_processed)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addLayout(btn_layout)

        self.setLayout(v_layout)

    def start(self):
        self.timer.start(30)  # Update every 30 ms

    def stop(self):
        self.timer.stop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 480))

        # Process frame with YOLO
        results = self.model.predict(source=frame, conf=0.25, save=False, show=False)
        processed_frame = results[0].plot()

        # Convert frames to Qt format
        unprocessed_qimg = self.convert_cv_qt(frame)
        processed_qimg = self.convert_cv_qt(processed_frame)

        # Update image labels
        self.label_unprocessed.setPixmap(unprocessed_qimg)
        self.label_processed.setPixmap(processed_qimg)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image).scaled(
            320, 240, Qt.KeepAspectRatio
        )  # Adjust size as needed
        return pixmap

    def closeEvent(self, event):
        """Handle the window close event."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
