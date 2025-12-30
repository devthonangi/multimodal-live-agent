# temporal_memory.py
class TemporalMemory:
    def __init__(self):
        self.prev_frame_objects = set()

    def detect_changes(self, current_objects):
        current_set = set(current_objects)
        appeared = current_set - self.prev_frame_objects
        disappeared = self.prev_frame_objects - current_set
        self.prev_frame_objects = current_set
        return appeared, disappeared
