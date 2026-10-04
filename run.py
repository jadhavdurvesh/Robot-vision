from __future__ import annotations

import argparse
import os

from app import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=["auto", "local", "cloud"], default=os.getenv("ROBOT_VISION_BACKEND", "auto"))
    parser.add_argument("--cloud-url", default=os.getenv("ROBOT_VISION_CLOUD_URL", ""))
    known, remaining = parser.parse_known_args()

    # Backend selection is intentionally session-based. Nothing runs until
    # this command is started, and everything stops when the process exits.
    os.environ["ROBOT_VISION_SELECTED_BACKEND"] = known.backend
    os.environ["ROBOT_VISION_CLOUD_URL"] = known.cloud_url

    main(remaining)
