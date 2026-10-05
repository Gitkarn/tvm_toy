"""2단계: 같은 계산을 TensorIR로 표현하고, 루프를 바꾼 뒤 CPU에서 실행."""

import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T

from common import expected, inputs, save_ir


@I.ir_module
class MatmulRelu:
    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((4, 4), "float32"),
        B: T.Buffer((4, 4), "float32"),
        C: T.Buffer((4, 4), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        with T.sblock("root"):
            T.reads()
            T.writes()
            Y = T.sblock_alloc_buffer((4, 4), "float32")
            for i, j, k in T.grid(4, 4, 4):
                with T.sblock("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] += A[vi, vk] * B[vk, vj]
            for i, j in T.grid(4, 4):
                with T.sblock("relu"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


def main() -> None:
    a, b = inputs()
    save_ir("01_tensorir_original.py", MatmulRelu.script())
    # BREAKPOINT 3: MatmulRelu.script()를 Debug Console에서 확인
    schedule = tvm.s_tir.Schedule(MatmulRelu)
    block = schedule.get_sblock("matmul")
    i, j, k = schedule.get_loops(block)
    # BREAKPOINT 4: 아래 한 줄 실행 전후의 schedule.mod.script() 비교
    j_outer, j_inner = schedule.split(j, factors=[None, 2])
    save_ir("02_tensorir_scheduled.py", schedule.mod.script())
    save_ir("03_schedule_trace.txt", str(schedule.trace))
    print("분할된 루프:", j_outer, j_inner, "원래 i/k:", i, k)

    # BREAKPOINT 5: 여기서부터 IR이 LLVM 대상 코드로 컴파일됨
    lib = tvm.tirx.build(schedule.mod, target="llvm")
    print("컴파일된 모듈:", lib.kind, "imports:", [item.kind for item in lib.imports])

    a_tvm = tvm.runtime.tensor(a)
    b_tvm = tvm.runtime.tensor(b)
    c_tvm = tvm.runtime.tensor(np.zeros((4, 4), dtype="float32"))
    # BREAKPOINT 6: TVM 런타임이 컴파일한 CPU 함수를 호출
    lib(a_tvm, b_tvm, c_tvm)
    result = c_tvm.numpy()
    np.testing.assert_allclose(result, expected(a, b), rtol=1e-6)
    print("TensorIR 결과:\n", result)


if __name__ == "__main__":
    main()
