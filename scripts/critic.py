class Critic:
    """Keeps captions safe for rendering without changing detector facts."""

    def __init__(self, max_length=180):
        self.max_length = max_length

    def evaluate(self, reasoning, detected_objects=None):
        caption = " ".join(reasoning.split())
        if len(caption) <= self.max_length:
            return caption
        return caption[: self.max_length - 1].rstrip() + "…"
