from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from critic import Critic
from llava_reasoner import SceneReasoner
from memory import Memory
from temporal_memory import TemporalMemory
from vision import DEFAULT_MODEL_PATH, Vision


@dataclass
class AgentConfig:
    source: int | str = 0
    model: Path = DEFAULT_MODEL_PATH
    confidence: float = 0.35
    memory_window: int = 10
    stability_frames: int = 2
    motion_threshold: float = 0.012
    output: Path | None = None
    display: bool = True
    max_frames: int | None = None
    device: str | None = None


class Agent:
    def __init__(self, config=None):
        self.config = config or AgentConfig()
        self.memory = Memory(window=self.config.memory_window)
        self.temporal = TemporalMemory(self.config.stability_frames)
        self.vision = Vision(self.config.model, self.config.confidence, self.config.device)
        self.reasoner = SceneReasoner(self.config.motion_threshold)
        self.critic = Critic()
        self.frame_index = 0

    def run(self):
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.source}")

        writer = None
        started = perf_counter()
        print(f"[Agent] Started source={self.config.source}")
        try:
            while self.config.max_frames is None or self.frame_index < self.config.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = self.vision.analyze(frame)
                objects = [detection.label for detection in detections]
                self.memory.update(objects, self.frame_index)
                reasoning = self.reasoner.reason(frame, objects, self.temporal)
                caption = self.critic.evaluate(reasoning, objects)

                elapsed = max(perf_counter() - started, 1e-9)
                fps = (self.frame_index + 1) / elapsed
                self.vision.draw(frame, detections)
                self._draw_hud(frame, caption, fps)

                if self.config.output:
                    writer = writer or self._create_writer(cap, frame)
                    writer.write(frame)
                self.frame_index += 1
                if self.config.display:
                    cv2.imshow("Multimodal Live Agent", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.config.display:
                cv2.destroyAllWindows()

        elapsed = max(perf_counter() - started, 1e-9)
        stats = {"frames": self.frame_index, "elapsed": elapsed, "fps": self.frame_index / elapsed}
        print(f"[Agent] Stopped frames={stats['frames']} average_fps={stats['fps']:.1f}")
        return stats

    def _create_writer(self, cap, frame):
        output = Path(self.config.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output}")
        return writer

    @staticmethod
    def _draw_hud(frame, caption, fps):
        lines = [caption[index : index + 80] for index in range(0, len(caption), 80)]
        overlay_height = 35 + 25 * len(lines)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (10, 50 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
