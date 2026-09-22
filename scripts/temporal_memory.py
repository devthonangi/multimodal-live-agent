class TemporalMemory:
    """Tracks scene transitions while suppressing one-frame detection flicker."""

    def __init__(self, stability_frames=2):
        if stability_frames < 1:
            raise ValueError("stability_frames must be at least 1")
        self.stability_frames = stability_frames
        self.stable_objects = set()
        self._candidate_objects = set()
        self._candidate_age = 0

    def detect_changes(self, current_objects):
        current_set = set(current_objects)
        if current_set != self._candidate_objects:
            self._candidate_objects = current_set
            self._candidate_age = 1
        else:
            self._candidate_age += 1
        if self._candidate_age < self.stability_frames:
            return set(), set()

        appeared = current_set - self.stable_objects
        disappeared = self.stable_objects - current_set
        self.stable_objects = current_set
        return appeared, disappeared
