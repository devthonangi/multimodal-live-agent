from collections import Counter

import cv2


class SceneReasoner:
    """Produces deterministic scene descriptions and frame-motion estimates."""

    def __init__(self, motion_threshold=0.012):
        self.motion_threshold = motion_threshold
        self.prev_frame_gray = None

    def reason(self, frame, objects, temporal_memory=None):
        counts = Counter(objects)
        if counts:
            items = [f"{count} {label}" for label, count in sorted(counts.items())]
            parts = ["Scene: " + ", ".join(items)]
        else:
            parts = ["No objects detected"]

        if temporal_memory:
            appeared, disappeared = temporal_memory.detect_changes(objects)
            if appeared:
                parts.append("Entered: " + ", ".join(sorted(appeared)))
            if disappeared:
                parts.append("Left: " + ", ".join(sorted(disappeared)))

        motion_score = self._motion_score(frame)
        if motion_score >= self.motion_threshold:
            parts.append(f"Motion: {motion_score:.1%}")
        if "person" in counts:
            parts.append("People present")
        return " | ".join(parts)

    def _motion_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        score = 0.0
        if self.prev_frame_gray is not None:
            difference = cv2.absdiff(self.prev_frame_gray, gray)
            _, changed = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
            score = cv2.countNonZero(changed) / changed.size
        self.prev_frame_gray = gray
        return score


SimpleLLaVA = SceneReasoner
