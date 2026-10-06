from __future__ import annotations

import argparse
from pathlib import Path

from vision.calibration import calibrate_checkerboard, save_calibration


def main():
    parser = argparse.ArgumentParser(description="Calibrate the Robot Vision phone camera")
    parser.add_argument("images", nargs="+", help="checkerboard image files")
    parser.add_argument("--output", default="calibration/camera.json")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=0.024,
                        help="checkerboard square side in metres")
    args = parser.parse_args()

    result = calibrate_checkerboard(
        args.images,
        board_size=(args.cols, args.rows),
        square_size=args.square_size,
    )
    save_calibration(args.output, result)
    print(f"Saved calibration: {Path(args.output).resolve()}")
    print(f"RMS reprojection error: {result['rms_error']:.4f}")
    print(f"Image size: {result['image_width']}x{result['image_height']}")


if __name__ == "__main__":
    main()
