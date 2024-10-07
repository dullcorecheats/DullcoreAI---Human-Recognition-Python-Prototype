import sys
import cv2
import numpy as np
import mss
import pyautogui
import keyboard  # For detecting keypress events
from tkinter import Tk, Label, Button, StringVar, Entry
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtCore, QtGui
import threading

class TransparentOverlay(QtWidgets.QMainWindow):
    def __init__(self, model, keys_config):
        super().__init__()
        self.model = model

        # Set up transparent window
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.X11BypassWindowManagerHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(30)  # Refresh every 30ms

        screen_resolution = QtWidgets.QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen_resolution.width(), screen_resolution.height())

        # Initialize screen capture with mss
        self.sct = mss.mss()

        # Initialize an empty list for bounding boxes
        self.bounding_boxes = []

        # State for snapping
        self.snap_body = False
        self.snap_head = False

        # Key bindings
        self.keys_config = keys_config  # {'snap_body': 'e', 'snap_head': 'r'}

        # Add event listeners for key press (handled via the PyQt5 timer)
        self.timer.timeout.connect(self.check_keypress)

    def update_overlay(self):
        # Capture screen using mss
        monitor = self.sct.monitors[1]  # Get the primary monitor
        screen = np.array(self.sct.grab(monitor))

        # Convert the BGRA (from mss) to RGB
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

        # YOLO human detection
        results = self.model(screen)
        
        # Clear and prepare for painting bounding boxes
        self.bounding_boxes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = box.cls
                if cls == 0:  # Class 0 is for humans in YOLO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Full-body bounding box

                    # Add full-body bounding box to list
                    self.bounding_boxes.append((x1, y1, x2, y2, 'body'))

                    # Estimate head region
                    body_height = y2 - y1
                    body_width = x2 - x1

                    # Approximate head height as 1/8 of the body height
                    head_height = int(body_height * 0.125)

                    # Approximate head width as 1/4 of the body width
                    head_width = int(body_width * 0.25)

                    # Calculate the head's top-left and bottom-right coordinates
                    head_x1 = x1 + (body_width // 2) - (head_width // 2)
                    head_x2 = head_x1 + head_width
                    head_y2 = y1 + head_height

                    # Add head bounding box to list
                    self.bounding_boxes.append((head_x1, y1, head_x2, head_y2, 'head'))

                    # Handle snapping
                    if self.snap_body:
                        body_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        self.move_mouse(body_center)
                        self.snap_body = False  # Snap only once per detection

                    if self.snap_head:
                        head_center = ((head_x1 + head_x2) // 2, (y1 + head_y2) // 2)
                        self.move_mouse(head_center)
                        self.snap_head = False  # Snap only once per detection

        # Trigger the paint event to draw bounding boxes
        self.repaint()

    def move_mouse(self, position):
        # Move mouse to the specified (x, y) position
        pyautogui.moveTo(position[0], position[1])

    def check_keypress(self):
        # Check for keypress based on current configuration
        if keyboard.is_pressed(self.keys_config['snap_body']):
            self.snap_body = True
        if keyboard.is_pressed(self.keys_config['snap_head']):
            self.snap_head = True

    def draw_bounding_box(self, painter, x1, y1, x2, y2, label):
        # Set different colors for head and body boxes
        if label == 'body':
            color = QtCore.Qt.red  # Red for body
        else:
            color = QtCore.Qt.blue  # Blue for head
        
        painter.setPen(QtGui.QPen(color, 3))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def paintEvent(self, event):
        # Draw bounding boxes on the transparent overlay
        painter = QtGui.QPainter(self)
        for (x1, y1, x2, y2, label) in self.bounding_boxes:
            self.draw_bounding_box(painter, x1, y1, x2, y2, label)

class SettingsWindow(Tk):
    def __init__(self, keys_config):
        super().__init__()
        self.keys_config = keys_config
        self.title("Settings")
        self.geometry("300x200")

        # Label and entry for snapping to body key
        Label(self, text="Snap to Body Key:").pack(pady=5)
        self.body_key_entry = StringVar(value=self.keys_config['snap_body'])
        Entry(self, textvariable=self.body_key_entry).pack(pady=5)

        # Label and entry for snapping to head key
        Label(self, text="Snap to Head Key:").pack(pady=5)
        self.head_key_entry = StringVar(value=self.keys_config['snap_head'])
        Entry(self, textvariable=self.head_key_entry).pack(pady=5)

        # Save button
        Button(self, text="Save", command=self.save_keys).pack(pady=20)

    def save_keys(self):
        self.keys_config['snap_body'] = self.body_key_entry.get()
        self.keys_config['snap_head'] = self.head_key_entry.get()
        print(f"Updated Key Bindings: {self.keys_config}")
        self.destroy()  # Close the settings window

def run_overlay(model, keys_config):
    # Start the overlay in a separate thread
    app = QtWidgets.QApplication(sys.argv)
    overlay = TransparentOverlay(model, keys_config)
    overlay.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use YOLOv8 model with pretrained weights

    # Create keys configuration
    keys_config = {'snap_body': 'e', 'snap_head': 'r'}  # Default keys

    # Create and run settings window
    settings_window = SettingsWindow(keys_config)
    settings_window.mainloop()

    # Run overlay after settings window is closed
    run_overlay(model, keys_config)
