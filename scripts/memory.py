from collections import Counter, deque


class Memory:
    """Bounded scene history with current and recent object summaries."""

    def __init__(self, window=5):
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, objects, frame_index):
        self.history.append({"frame": frame_index, "objects": list(objects)})

    def summary(self):
        return list(self.history[-1]["objects"]) if self.history else []

    def counts(self):
        return Counter(self.summary())

    def recent_objects(self):
        return sorted({obj for item in self.history for obj in item["objects"]})

    def full_summary(self):
        return list(self.history)
