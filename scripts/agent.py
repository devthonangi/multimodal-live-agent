# agent.py
import cv2
from memory import Memory
from temporal_memory import TemporalMemory
from vision import Vision
from llava_reasoner import SimpleLLaVA
from critic import Critic

class Agent:
    def __init__(self):
        self.memory = Memory(window=5)
        self.temporal = TemporalMemory()
        self.vision = Vision()
        self.reasoner = SimpleLLaVA()  # motion-aware reasoning
        self.critic = Critic()
        self.frame_index = 0

    def run(self):
        print("[Agent] Multimodal Agent Started")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Detect objects
            objects = self.vision.analyze(frame)

            # Update memory
            self.memory.update(objects, self.frame_index)
            summary = self.memory.summary()

            # Live reasoning with motion
            reasoning = self.reasoner.reason(frame, summary, self.temporal)

            # Critic checks reasoning
            final_caption = self.critic.evaluate(reasoning, summary)

            # Display on frame
            cv2.putText(frame, final_caption, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Multimodal Agent", frame)

            self.frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[Agent] Agent Stopped")
