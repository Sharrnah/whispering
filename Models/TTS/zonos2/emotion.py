"""Official ZONOS2 projected-speaker emotion direction support.

The released vectors are additive directions in the model's projected 2048-D
speaker space. Loading and calibration mirror Zyphra/ZONOS2's dependency-light
emotion implementation, restricted to the released ``space=proj`` format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


@dataclass
class EmotionDirections:
    named: dict[str, torch.Tensor]
    axes: dict[str, torch.Tensor]
    calibration: dict[str, float]
    global_calibration: float
    dim: int
    space: str

    @classmethod
    def load(cls, directory: Path) -> "EmotionDirections | None":
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dim = int(manifest["dim"])
        space = str(manifest.get("space", "raw"))
        if space != "proj":
            raise ValueError(
                f"Whispering Tiger expects projected ZONOS2 emotion directions, got {space!r}."
            )

        named: dict[str, torch.Tensor] = {}
        axes: dict[str, torch.Tensor] = {}
        for name, entry in manifest.get("directions", {}).items():
            path = directory / entry["file"]
            array = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            if array.shape[0] != dim:
                raise ValueError(
                    f"ZONOS2 emotion direction {name!r} has {array.shape[0]} values; expected {dim}."
                )
            vector = torch.from_numpy(np.ascontiguousarray(array)).float().cpu()
            if str(entry.get("kind")) == "axis" or name in {"valence", "arousal"}:
                axes[name] = vector
            else:
                named[name] = vector

        calibration_path = directory / "calibration.json"
        calibration: dict[str, float] = {}
        global_calibration = 1.0
        if calibration_path.is_file():
            raw = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration = {
                str(name): float(value)
                for name, value in raw.get("default", {}).items()
            }
            global_calibration = float(raw.get("global_default", 1.0))
        return cls(named, axes, calibration, global_calibration, dim, space)

    def calibrated_strength(self, name: str) -> float:
        return self.calibration.get(name, self.global_calibration)


def emotion_hidden_delta(
    directions: EmotionDirections | None,
    *,
    sliders: Mapping[str, float],
    valence: float,
    arousal: float,
    strength: float,
) -> torch.Tensor | None:
    if directions is None or strength == 0.0:
        return None
    delta = torch.zeros(directions.dim, dtype=torch.float32)
    requested = False
    for name, weight in sliders.items():
        weight = float(weight)
        if weight == 0.0:
            continue
        vector = directions.named.get(name)
        if vector is None:
            raise ValueError(f"Unknown ZONOS2 emotion direction: {name}")
        delta.add_(vector, alpha=weight * directions.calibrated_strength(name))
        requested = True
    for name, weight in (("valence", valence), ("arousal", arousal)):
        weight = float(weight)
        if weight == 0.0:
            continue
        vector = directions.axes.get(name)
        if vector is None:
            raise ValueError(f"ZONOS2 emotion axis is unavailable: {name}")
        delta.add_(vector, alpha=weight * directions.calibrated_strength(name))
        requested = True
    if not requested:
        return None
    return delta.mul(float(strength)).contiguous()

