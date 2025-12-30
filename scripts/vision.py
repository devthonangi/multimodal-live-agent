# vision.py
from ultralytics import YOLO

class Vision:
    def __init__(self):
        self.model = YOLO("yolov8m.pt")  # Place yolov8m.pt in project root

    def analyze(self, frame):
        results = self.model(frame, verbose=False)[0]
        objects = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            objects.append(cls_name)
        return objects
