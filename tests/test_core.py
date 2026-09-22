import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from critic import Critic
from llava_reasoner import SceneReasoner
from memory import Memory
from run import parse_source
from temporal_memory import TemporalMemory


class MemoryTests(unittest.TestCase):
    def test_window_and_counts(self):
        memory = Memory(window=2)
        memory.update(["person", "person"], 0)
        memory.update(["bottle"], 1)
        memory.update(["chair"], 2)
        self.assertEqual(len(memory.full_summary()), 2)
        self.assertEqual(memory.counts()["chair"], 1)
        self.assertEqual(memory.recent_objects(), ["bottle", "chair"])


class TemporalMemoryTests(unittest.TestCase):
    def test_stable_transitions(self):
        memory = TemporalMemory(stability_frames=2)
        self.assertEqual(memory.detect_changes(["person"]), (set(), set()))
        self.assertEqual(memory.detect_changes(["person"]), ({"person"}, set()))
        self.assertEqual(memory.detect_changes([]), (set(), set()))
        self.assertEqual(memory.detect_changes([]), (set(), {"person"}))


class ReasonerTests(unittest.TestCase):
    def test_counts_motion_and_caption_limit(self):
        reasoner = SceneReasoner(motion_threshold=0.01)
        still = np.zeros((100, 100, 3), dtype=np.uint8)
        changed = still.copy()
        cv2.rectangle(changed, (10, 10), (90, 90), (255, 255, 255), -1)
        first = reasoner.reason(still, ["person", "person"])
        second = reasoner.reason(changed, ["person"])
        self.assertIn("2 person", first)
        self.assertIn("Motion:", second)
        self.assertEqual(len(Critic(max_length=20).evaluate("x" * 50)), 20)

    def test_source_parsing(self):
        self.assertEqual(parse_source("1"), 1)
        self.assertEqual(parse_source("video.mp4"), "video.mp4")


if __name__ == "__main__":
    unittest.main()
