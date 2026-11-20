#!/usr/bin/env python3
"""Create review-only short clips and SRT captions from a local video you own or may use.

This script does not search YouTube, download third-party videos, or upload anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def probe_duration(path: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError("The source video has no valid duration.")
    return duration


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def write_srt(segments: list[dict], destination: Path) -> None:
    lines: list[str] = []
    index = 1
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, float(segment["start"]))
        end = max(start + 0.1, float(segment["end"]))
        lines.extend([str(index), f"{timestamp(start)} --> {timestamp(end)}", text, ""])
        index += 1
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to an authorized local video")
    parser.add_argument("--output", type=Path, default=Path("clips"))
    parser.add_argument("--clip-seconds", type=int, default=30)
    parser.add_argument("--max-clips", type=int, default=5)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"Input video does not exist: {source}")
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        parser.error("Input must be a supported local video file (mp4, mov, mkv, webm, or m4v).")
    if not 5 <= args.clip_seconds <= 180:
        parser.error("--clip-seconds must be between 5 and 180.")
    if not 1 <= args.max_clips <= 20:
        parser.error("--max-clips must be between 1 and 20.")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        parser.error("ffmpeg and ffprobe must be installed and available on PATH.")

    try:
        import whisper
    except ImportError:
        parser.error("Install Whisper first with: pip install openai-whisper")

    output.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(source)
    model = whisper.load_model("tiny")
    clip_count = min(args.max_clips, int(duration // args.clip_seconds))
    if clip_count == 0:
        parser.error(f"Video must be at least {args.clip_seconds} seconds long.")

    print(f"Creating {clip_count} review clip(s) from {source.name}...")
    for number in range(clip_count):
        start = number * args.clip_seconds
        clip = output / f"clip-{number + 1:02}.mp4"
        captions = output / f"clip-{number + 1:02}.srt"
        run([
            "ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(args.clip_seconds),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
            "-b:a", "128k", "-movflags", "+faststart", str(clip),
        ])
        transcription = model.transcribe(str(clip), fp16=False, verbose=False)
        write_srt(transcription.get("segments", []), captions)
        print(f"Created {clip.name} and {captions.name}")

    print(f"Done. Review the files in {output}; nothing was uploaded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        print(f"Command failed: {detail}", file=sys.stderr)
        raise SystemExit(error.returncode)
