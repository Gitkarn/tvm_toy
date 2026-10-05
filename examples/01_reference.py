"""1단계: TVM 없이 y = relu(A @ B)를 Python 문장으로 따라가기."""

import numpy as np

from common import expected, inputs


def matmul_relu_python(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    out = np.zeros((n, n), dtype="float32")
    for i in range(n):
        for j in range(n):
            acc = np.float32(0)
            for k in range(n):
                # BREAKPOINT 1: i, j, k, acc, a[i, k], b[k, j]를 Watch에 추가
                acc += a[i, k] * b[k, j]
            # BREAKPOINT 2: 누산 결과와 ReLU 후 값을 비교
            out[i, j] = max(acc, 0)
    return out


def main() -> None:
    a, b = inputs()
    result = matmul_relu_python(a, b)
    np.testing.assert_allclose(result, expected(a, b), rtol=1e-6)
    print("Python 결과:\n", result)


if __name__ == "__main__":
    main()
