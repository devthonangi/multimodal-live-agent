# llava_reasoner.py
import cv2
import numpy as np

class SimpleLLaVA:
    """Live scene reasoning with temporal awareness and motion detection"""
    def __init__(self):
        print("[LLaVA] Reasoner initialized")
        self.prev_frame_gray = None

    def reason(self, frame, memory_summary, temporal_memory=None):
        if not memory_summary:
            return "No objects detected."

        # Count objects
        counts = {}
        for obj in memory_summary:
            counts[obj] = counts.get(obj, 0) + 1

        # Build descriptive phrases
        phrases = [f"{count} {obj}(s)" for obj, count in counts.items()]
        caption = "Scene contains: " + ", ".join(phrases)

        # Add temporal info
        if temporal_memory:
            appeared, disappeared = temporal_memory.detect_changes(memory_summary)
            if appeared:
                caption += " | New objects: " + ", ".join(appeared)
            if disappeared:
                caption += " | Objects left: " + ", ".join(disappeared)

        # Motion detection
        motion_phrases = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame_gray is not None:
            frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            motion_area = cv2.countNonZero(thresh)

            if motion_area > 500:  # tweak threshold for sensitivity
                if "person" in memory_summary:
                    motion_phrases.append("Person is moving")
                if "bottle" in memory_summary:
                    motion_phrases.append("Bottle moved")

        self.prev_frame_gray = gray

        if motion_phrases:
            caption += " | " + ", ".join(motion_phrases)

        # Basic presence info
        if "person" in memory_summary:
            caption += " | People are present."

        return caption
