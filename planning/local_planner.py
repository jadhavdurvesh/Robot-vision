from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlannedPath:
    points: list[tuple[int, int]]
    direction: str
    clear: bool


class LocalPlanner:
    """Choose an image-space corridor toward the forward direction."""

    def __init__(self, samples: int, top_ratio: float, bottom_ratio: float, lookahead_ratio: float, margin: int) -> None:
        self.samples = max(5, samples)
        self.top_ratio = top_ratio
        self.bottom_ratio = bottom_ratio
        self.lookahead_ratio = lookahead_ratio
        self.margin = max(0, margin)

    def plan(self, free_mask: np.ndarray) -> PlannedPath:
        height, width = free_mask.shape
        y_bottom = min(height - 1, int(height * self.bottom_ratio))
        y_top = max(0, int(height * self.top_ratio))
        center = width // 2

        candidates = np.linspace(int(width * 0.12), int(width * 0.88), self.samples).astype(int)
        best_x = center
        best_score = -1.0
        best_ratio = 0.0

        for x in candidates:
            x1 = max(0, x - self.margin)
            x2 = min(width, x + self.margin + 1)
            roi = free_mask[y_top:y_bottom + 1, x1:x2]
            if roi.size == 0:
                continue
            ratio = float(np.count_nonzero(roi)) / float(roi.size)
            center_penalty = abs(x - center) / max(1, center)
            score = ratio - 0.18 * center_penalty
            if score > best_score:
                best_score = score
                best_x = int(x)
                best_ratio = ratio

        if best_ratio > 0.55:
            target_y = int(height * self.lookahead_ratio)
            points = [
                (center, y_bottom),
                (int((center + best_x) / 2), int((y_bottom + target_y) / 2)),
                (best_x, target_y),
                (best_x, y_top),
            ]
            direction = "FORWARD" if abs(best_x - center) < width * 0.08 else ("LEFT" if best_x < center else "RIGHT")
            return PlannedPath(points=points, direction=direction, clear=True)

        return PlannedPath(points=[(center, y_bottom)], direction="BLOCKED", clear=False)
