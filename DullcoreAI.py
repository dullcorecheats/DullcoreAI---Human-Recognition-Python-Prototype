import sys
import cv2
import numpy as np
import mss
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080} 
sct = mss.mss()

class SettingsDialog(QDialog):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.setWindowTitle("Settings")

        layout = QVBoxLayout()

        conf_label = QLabel("Detection Confidence")
        layout.addWidget(conf_label)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.overlay.detection_confidence * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        layout.addWidget(self.conf_slider)

        line_thickness_label = QLabel("Line Thickness")
        layout.addWidget(line_thickness_label)
        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setMinimum(1)
        self.line_thickness_slider.setMaximum(10)
        self.line_thickness_slider.setValue(self.overlay.line_thickness)
        layout.addWidget(self.line_thickness_slider)

        lod_label = QLabel("Level of Detail")
        layout.addWidget(lod_label)
        self.lod_combo = QComboBox()
        self.lod_combo.addItem("Low (Bounding Box Only)")
        self.lod_combo.addItem("Medium (Partial Skeleton)")
        self.lod_combo.addItem("High (Full Skeleton)")
        self.lod_combo.setCurrentIndex(self.overlay.level_of_detail)  
        layout.addWidget(self.lod_combo)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def apply_settings(self):
        self.overlay.detection_confidence = self.conf_slider.value() / 100
        self.overlay.line_thickness = self.line_thickness_slider.value()
        self.overlay.level_of_detail = self.lod_combo.currentIndex()  
        self.overlay.update_pose_object()
        self.accept()

class Overlay(QLabel):
    def __init__(self):
        super().__init__()

        self.detection_confidence = 0.6
        self.line_thickness = 2
        self.level_of_detail = 2  

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.setGeometry(0, 0, monitor["width"], monitor["height"])

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(30) 

        self.pose = mp_pose.Pose(min_detection_confidence=self.detection_confidence, min_tracking_confidence=0.6)

        self.open_settings_menu()

    def update_pose_object(self):
        self.pose = mp_pose.Pose(min_detection_confidence=self.detection_confidence, min_tracking_confidence=0.6)

    def open_settings_menu(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def update_overlay(self):
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        overlay_image = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        overlay_image.fill(Qt.transparent)

        if results.pose_landmarks:
            height, width, _ = frame.shape


            painter = QPainter(overlay_image)


            self.draw_bounding_box(painter, results.pose_landmarks, width, height)

            if self.level_of_detail == 2:  
                self.draw_skeleton(painter, results.pose_landmarks, width, height)
            elif self.level_of_detail == 1:  
                self.draw_partial_skeleton(painter, results.pose_landmarks, width, height)

            painter.end()

        self.setPixmap(QPixmap.fromImage(overlay_image))

    def draw_skeleton(self, painter, pose_landmarks, width, height):
        """Draw the full pose skeleton using QPainter."""
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = pose_landmarks.landmark[start_idx]
            end_landmark = pose_landmarks.landmark[end_idx]

            start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
            end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

            painter.setPen(QPen(QColor(0, 255, 0), self.line_thickness, Qt.SolidLine))  
            painter.drawLine(start_point[0], start_point[1], end_point[0], end_point[1])

    def draw_partial_skeleton(self, painter, pose_landmarks, width, height):
        """Draw a partial pose skeleton (e.g., only the torso and limbs)."""
        partial_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)
        ]
        for start_idx, end_idx in partial_connections:
            start_landmark = pose_landmarks.landmark[start_idx]
            end_landmark = pose_landmarks.landmark[end_idx]

            # Scale landmarks to screen size
            start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
            end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

            painter.setPen(QPen(QColor(0, 255, 0), self.line_thickness, Qt.SolidLine))  
            painter.drawLine(start_point[0], start_point[1], end_point[0], end_point[1])

    def draw_bounding_box(self, painter, pose_landmarks, width, height):
        """Draws a bounding box around the detected pose."""
        landmarks = pose_landmarks.landmark
        x_min = min(int(landmark.x * width) for landmark in landmarks)
        x_max = max(int(landmark.x * width) for landmark in landmarks)
        y_min = min(int(landmark.y * height) for landmark in landmarks)
        y_max = max(int(landmark.y * height) for landmark in landmarks)

        painter.setPen(QPen(QColor(255, 0, 0), self.line_thickness, Qt.SolidLine))  
        painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
