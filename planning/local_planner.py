from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlannedPath:
    points: list[tuple[int, int]]
    direction: str
    clear: bool
    clearance: float = 0.0
    target_x: int = 0


class LocalPlanner:
    """Stable image-space local planner with corridor width and temporal hysteresis."""

    def __init__(
        self,
        samples: int,
        top_ratio: float,
        bottom_ratio: float,
        lookahead_ratio: float,
        margin: int,
    ) -> None:
        self.samples = max(9, samples)
        self.top_ratio = top_ratio
        self.bottom_ratio = bottom_ratio
        self.lookahead_ratio = lookahead_ratio
        self.margin = max(4, margin)
        self._smoothed_x: float | None = None
        self._stable_direction = "FORWARD"
        self._direction_votes: dict[str, int] = {"LEFT": 0, "RIGHT": 0, "FORWARD": 0}

    def reset(self) -> None:
        self._smoothed_x = None
        self._stable_direction = "FORWARD"
        self._direction_votes = {"LEFT": 0, "RIGHT": 0, "FORWARD": 0}

    def plan(self, free_mask: np.ndarray) -> PlannedPath:
        height, width = free_mask.shape
        y_bottom = min(height - 1, int(height * self.bottom_ratio))
        y_top = max(0, int(height * self.top_ratio))
        center = width // 2

        # Light morphological cleanup prevents one-pixel holes from creating
        # fake corridors while retaining real narrow openings.
        region = free_mask[y_top:y_bottom + 1, :]
        if region.size:
            region = cv2.morphologyEx(
                region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
            )
            region = cv2.morphologyEx(
                region, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
            )

        candidates = np.linspace(
            int(width * 0.10), int(width * 0.90), self.samples
        ).astype(int)

        best_x = center
        best_score = -1e9
        best_ratio = 0.0
        best_clearance = 0.0

        for x in candidates:
            half = max(self.margin, int(width * 0.035))
            x1 = max(0, x - half)
            x2 = min(width, x + half + 1)

            if region.size == 0:
                continue

            corridor = region[:, x1:x2]
            ratio = float(np.count_nonzero(corridor)) / float(corridor.size)

            # Estimate the narrowest horizontal free width through several
            # lookahead depths. This discourages paths that are technically
            # open but squeeze through a tiny gap.
            rows = np.linspace(0, max(0, region.shape[0] - 1), 7).astype(int)
            widths = []
            for row in rows:
                free_row = region[row] > 0
                left = x
                while left > 0 and free_row[left - 1]:
                    left -= 1
                right = x
                while right < width - 1 and free_row[right + 1]:
                    right += 1
                widths.append((right - left + 1) / max(1, width))

            clearance = float(min(widths)) if widths else 0.0
            center_penalty = abs(x - center) / max(1, center)

            score = (
                0.52 * ratio
                + 0.38 * clearance
                - 0.14 * center_penalty
            )

            if score > best_score:
                best_score = score
                best_x = int(x)
                best_ratio = ratio
                best_clearance = clearance

        if best_ratio <= 0.42 or best_clearance < 0.025:
            self._direction_votes["FORWARD"] = 0
            return PlannedPath(
                points=[(center, y_bottom)],
                direction="BLOCKED",
                clear=False,
                clearance=best_clearance,
                target_x=center,
            )

        # Temporal smoothing: don't jump the path target wildly because of
        # one noisy detector frame.
        if self._smoothed_x is None:
            self._smoothed_x = float(best_x)
        else:
            self._smoothed_x = 0.72 * self._smoothed_x + 0.28 * best_x

        target_x = int(np.clip(self._smoothed_x, width * 0.08, width * 0.92))
        delta = target_x - center

        proposed = (
            "FORWARD"
            if abs(delta) < width * 0.09
            else ("LEFT" if delta < 0 else "RIGHT")
        )

        # Direction hysteresis: require repeated evidence before switching
        # between lateral commands.
        for key in self._direction_votes:
            self._direction_votes[key] = max(0, self._direction_votes[key] - 1)
        self._direction_votes[proposed] += 3

        if self._direction_votes[proposed] >= 5:
            self._stable_direction = proposed

        target_y = int(height * self.lookahead_ratio)
        mid_x = int((center + target_x) / 2)
        mid_y = int((y_bottom + target_y) / 2)

        points = [
            (center, y_bottom),
            (mid_x, mid_y),
            (target_x, target_y),
            (target_x, y_top),
        ]

        return PlannedPath(
            points=points,
            direction=self._stable_direction,
            clear=True,
            clearance=best_clearance,
            target_x=target_x,
        )
