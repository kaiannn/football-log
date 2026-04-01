"""基于球衣区域 HSV 的粗分队 + 时序平滑。"""

from collections import defaultdict, deque
from typing import Deque, Dict

import cv2
import numpy as np


class TeamClassifier:
    def __init__(self, history_len: int = 8):
        self._history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=history_len))
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_blue = np.array([100, 70, 50])
        self.upper_blue = np.array([130, 255, 255])

    def instant_label(self, frame, bbox) -> str:
        x, y, w, h = bbox
        y_center = y + int(h * 0.25)
        h_analyze = int(h * 0.5)
        if y_center + h_analyze > frame.shape[0] or x + w > frame.shape[1] or x < 0 or y_center < 0:
            return "Unknown Player"
        patch = frame[y_center : y_center + h_analyze, x : x + w]
        if patch.size == 0:
            return "Unknown Player"

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_patch, self.lower_red1, self.upper_red1) | cv2.inRange(
            hsv_patch, self.lower_red2, self.upper_red2
        )
        blue_mask = cv2.inRange(hsv_patch, self.lower_blue, self.upper_blue)
        red_count = int(np.sum(red_mask > 0))
        blue_count = int(np.sum(blue_mask > 0))
        if red_count > blue_count and red_count > 50:
            return "Red Player"
        if blue_count > red_count and blue_count > 50:
            return "Blue Player"
        return "Unknown Player"

    def smooth_label(self, track_id: int, instant_label: str) -> str:
        if "Player" not in instant_label:
            return instant_label
        history = self._history[track_id]
        history.append(instant_label)
        red_votes = sum(1 for x in history if x == "Red Player")
        blue_votes = sum(1 for x in history if x == "Blue Player")
        if red_votes > blue_votes:
            return "Red Player"
        if blue_votes > red_votes:
            return "Blue Player"
        return instant_label
