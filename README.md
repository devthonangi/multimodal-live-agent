# Multimodal Live Agent

A local, real-time scene-understanding agent powered by YOLO and OpenCV. It detects and labels objects, remembers recent observations, suppresses one-frame detection flicker, reports scene transitions and motion, and can save an annotated video.

## Features

- Webcam, video-file, or stream input
- YOLO object detection with configurable confidence and device
- Bounding boxes and confidence labels
- Object counts and bounded scene memory
- Debounced “entered” and “left” events
- Frame-difference motion estimates
- Live FPS and scene-description overlay
- Headless processing and annotated MP4 recording
- Unit-tested memory, temporal, reasoning, and CLI behavior

This project uses deterministic, local scene reasoning. Despite the historical filename `llava_reasoner.py`, it does not download or run a LLaVA language model.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ultralytics downloads `yolov8n.pt` on first use if it is not already in the project root. Model weights are ignored by Git. You can also supply another model with `--model`.

## Run

Start the webcam interface:

```bash
python scripts/run.py
```

Press `q` to stop. On macOS, grant camera access when prompted.

Process a video and save the annotated result:

```bash
python scripts/run.py --source input.mp4 --headless --output output.mp4
```

Use Apple Silicon acceleration and a stricter detector threshold:

```bash
python scripts/run.py --device mps --confidence 0.5
```

Useful options:

```text
--source CAMERA_OR_VIDEO
--model PATH
--confidence 0.35
--memory-window 10
--stability-frames 2
--motion-threshold 0.012
--device cpu|mps|0
--output annotated.mp4
--headless
--max-frames N
```

## Test

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

## Architecture

- `scripts/vision.py`: inference, structured detections, and bounding boxes
- `scripts/memory.py`: bounded frame history and object summaries
- `scripts/temporal_memory.py`: stable scene-transition detection
- `scripts/llava_reasoner.py`: object counts and motion-aware descriptions
- `scripts/critic.py`: display-safe caption normalization
- `scripts/agent.py`: video pipeline, HUD, recording, and lifecycle
- `scripts/run.py`: command-line interface

All inference stays local after dependencies and model weights are available.
