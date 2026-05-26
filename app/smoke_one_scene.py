from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.engine import PlaybackEngine, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute one full RE10K scene for the RTGS playback app.")
    parser.add_argument("--scene", default="5aca87f95a9412c6")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--render-first", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = PlaybackEngine(device=args.device, output_dir=REPO_ROOT / "app" / "outputs" / "playback_smoke")
    manifest = engine.start(
        scene=args.scene,
        checkpoint_path=args.checkpoint_path,
        num_frames=args.num_frames,
        gap=args.gap,
        stride=args.stride,
        max_snapshots=args.max_snapshots,
    )
    if args.render_first:
        first = engine.render(0, {})
        manifest["first_render_index"] = first["index"]
        manifest["first_render_timestamp"] = first["timestamp"]
        manifest["first_render_png_data_url_prefix"] = first["image"][:32]
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
