# critic.py
class Critic:
    """Simple critic to validate and refine reasoning"""
    def evaluate(self, reasoning, detected_objects):
        if not detected_objects:
            return reasoning

        # Correct obvious overcounting (example heuristic)
        corrected_phrases = []
        for phrase in reasoning.split(","):
            for obj in detected_objects:
                if obj in phrase:
                    corrected_phrases.append(f"1 {obj}(s)")
        corrected_phrases = list(dict.fromkeys(corrected_phrases))  # remove duplicates

        final_caption = "Scene contains: " + ", ".join(corrected_phrases)

        # Keep temporal/motion info if present
        if "|" in reasoning:
            temporal_info = " | ".join(reasoning.split("|")[1:])
            final_caption += " | " + temporal_info

        return final_caption
