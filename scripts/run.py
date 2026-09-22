import argparse
from pathlib import Path

from agent import Agent, AgentConfig
from vision import DEFAULT_MODEL_PATH


def parse_source(value):
    return int(value) if value.isdigit() else value


def build_parser():
    parser = argparse.ArgumentParser(description="Real-time local scene understanding with YOLO")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection threshold (0-1)")
    parser.add_argument("--memory-window", type=int, default=10)
    parser.add_argument("--stability-frames", type=int, default=2)
    parser.add_argument("--motion-threshold", type=float, default=0.012)
    parser.add_argument("--device", help="Inference device, for example cpu, mps, or 0")
    parser.add_argument("--output", type=Path, help="Write annotated MP4 output")
    parser.add_argument("--headless", action="store_true", help="Run without a preview window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    return parser


def main():
    args = build_parser().parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    config = AgentConfig(
        source=parse_source(args.source),
        model=args.model,
        confidence=args.confidence,
        memory_window=args.memory_window,
        stability_frames=args.stability_frames,
        motion_threshold=args.motion_threshold,
        output=args.output,
        display=not args.headless,
        max_frames=args.max_frames,
        device=args.device,
    )
    try:
        Agent(config).run()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
