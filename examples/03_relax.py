"""3단계: 그래프 수준 Relax IR이 TensorIR로 내려가 VM에서 실행되는 흐름."""

import numpy as np
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R

from common import expected, inputs, save_ir


@I.ir_module
class Graph:
    @R.function
    def main(
        a: R.Tensor((4, 4), "float32"), b: R.Tensor((4, 4), "float32")
    ) -> R.Tensor((4, 4), "float32"):
        with R.dataflow():
            matmul = R.matmul(a, b)
            relu = R.nn.relu(matmul)
            R.output(relu)
        return relu


@tvm.instrument.pass_instrument
class PassRecorder:
    """각 Relax pass 후 바뀐 IR을 저장하고 VS Code 중단점을 받을 위치."""

    def __init__(self, initial_script: str):
        self.previous_script = initial_script
        self.change_count = 0

    def run_after_pass(self, mod, info) -> None:
        # BREAKPOINT 8: info.name, mod.script()를 조사하면 pass별 변경이 보임
        current_script = mod.script()
        changed = current_script != self.previous_script
        print(f"pass {info.name}: {'IR 변경' if changed else 'IR 동일'}")
        if changed:
            self.change_count += 1
            save_ir(f"pass_{self.change_count:02}.py", current_script)
            self.previous_script = current_script


def main() -> None:
    a, b = inputs()
    original_script = Graph.script()
    save_ir("05_relax_original.py", original_script)
    # BREAKPOINT 7: Relax에는 연산 관계가 있고, 구체적인 CPU 루프는 아직 없음
    with tvm.transform.PassContext(instruments=[PassRecorder(original_script)]):
        lowered = relax.get_pipeline("zero")(Graph)
    save_ir("06_relax_lowered.py", lowered.script())
    # BREAKPOINT 9: 05와 06을 비교하여 생성된 PrimFunc와 call_tir 찾기
    executable = tvm.compile(lowered, target="llvm")
    # BREAKPOINT 10: 컴파일된 VM 실행 파일과 실제 CPU 장치를 연결
    vm = relax.VirtualMachine(executable, tvm.cpu())
    a_tvm = tvm.runtime.tensor(a)
    b_tvm = tvm.runtime.tensor(b)
    # BREAKPOINT 11: VM이 그래프 명령을 수행하고 컴파일된 커널을 호출
    result = vm["main"](a_tvm, b_tvm).numpy()
    np.testing.assert_allclose(result, expected(a, b), rtol=1e-6)
    print("Relax VM 결과:\n", result)


if __name__ == "__main__":
    main()
