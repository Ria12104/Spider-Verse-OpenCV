"""
sparkle.py — Sparkle particle class.

A Sparkle is a small, fading coloured circle that is spawned at the pinch
point and drifts upward, shrinking as its lifetime decreases.
"""

import random

import cv2

from config import SPARKLE_LIFE, SPARKLE_RADIUS


class Sparkle:
    """A single sparkle particle."""

    # Amber / gold tone palette
    _COLORS = [
        (0, 215, 255),
        (0, 255, 255),
        (0, 180, 255),
    ]

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)
        self.vx        = random.uniform(-3, 3)
        self.vy        = random.uniform(-4, -1)   # drift upward
        self.life      = self.max_life = SPARKLE_LIFE
        self.color     = random.choice(self._COLORS)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Advance position by velocity and decrement lifetime."""
        self.x    += self.vx
        self.y    += self.vy
        self.life -= 1

    def is_alive(self) -> bool:
        """Return True while the particle still has remaining lifetime."""
        return self.life > 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, frame) -> None:
        """Draw the particle onto *frame* with opacity proportional to lifetime."""
        fade = self.life / self.max_life
        cv2.circle(
            frame,
            (int(self.x), int(self.y)),
            max(2, int(SPARKLE_RADIUS * fade)),
            tuple(int(c * fade) for c in self.color),
            -1,
        )
