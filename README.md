# Multimodal Live Agent

A **live AI agent** for understanding and reasoning about scenes in real-time using your webcam. This project demonstrates **multimodal perception, temporal memory, and live reasoning** with a lightweight, local setup.

---

## **Overview**

The agent combines:

1. **Vision (YOLOv8)** – Detects objects in the video stream (e.g., people, bottles, chairs).
2. **Memory** – Keeps track of recent frames to understand scene consistency.
3. **Temporal Memory** – Detects objects that **appear** or **disappear** across frames.
4. **Live Reasoning (LLaVA-inspired)** – Generates descriptive captions of the scene with:
   - Counts of objects
   - Appearance/disappearance of objects
   - Motion detection (e.g., moving person or object)
   - Basic contextual information (e.g., “People are present”)

---

## **Design & Logic**

### 1. **Vision Module (`vision.py`)**
- Uses **YOLOv8 medium model** for object detection.
- Detects and classifies objects frame by frame.
- Returns a list of objects per frame.

### 2. **Memory (`memory.py`)**
- Stores a **sliding window of recent frames**.
- Keeps track of **unique objects** in each frame.
- Provides summaries for reasoning and temporal analysis.

### 3. **Temporal Memory (`temporal_memory.py`)**
- Compares current frame objects with previous frame objects.
- Identifies:
  - **Newly appeared objects**
  - **Objects that have left**
- Provides this info to the reasoning module for more dynamic descriptions.

### 4. **Live Reasoner (`llava_reasoner.py`)**
- Receives the current frame and memory summaries.
- Counts objects and generates a **descriptive caption**:
  - `"Scene contains: 1 person(s), 2 bottle(s) | New objects: person | Objects left: bottle | Person is moving"`
- Integrates **motion detection**:
  - Detects movement using frame differences.
  - Adds context like `"Person is moving"` or `"Bottle moved"`.

### 5. **Agent (`agent.py`)**
- Coordinates all modules.
- Captures live video from the webcam using OpenCV.
- Passes each frame to:
  1. Vision for object detection
  2. Memory and temporal memory for tracking
  3. Reasoner for generating live captions
- Displays captions on the video feed in real-time.


