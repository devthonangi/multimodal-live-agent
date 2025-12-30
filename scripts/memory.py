# memory.py
class Memory:
    def __init__(self, window=5):
        self.window = window
        self.history = []

    def update(self, objects, frame_index):
        self.history.append({"frame": frame_index, "objects": list(set(objects))})
        if len(self.history) > self.window:
            self.history.pop(0)

    def summary(self):
        if not self.history:
            return []
        return self.history[-1]["objects"]

    def full_summary(self):
        return self.history
