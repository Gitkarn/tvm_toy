"""세 예제에서 사용하는 동일한 4x4 입력과 산출물 경로."""

from pathlib import Path

import numpy as np


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def inputs() -> tuple[np.ndarray, np.ndarray]:
    a = np.array(
        [[1, -2, 0, 3], [0, 1, 2, -1], [-1, 0, 1, 2], [2, 1, -1, 0]],
        dtype="float32",
    )
    b = np.array(
        [[1, 0, -1, 2], [2, -1, 0, 1], [0, 2, 1, -2], [-1, 1, 2, 0]],
        dtype="float32",
    )
    return a, b


def expected(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a @ b, 0)


def save_ir(name: str, ir: str) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(ir, encoding="utf-8")
    print(f"IR 저장: {path}")
    return path
