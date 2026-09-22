from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class Vision:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.35, device=None):
        model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.device = device

    def analyze(self, frame):
        result = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    @staticmethod
    def draw(frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            text = f"{detection.label} {detection.confidence:.0%}"
            cv2.putText(frame, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2)
        return frame
