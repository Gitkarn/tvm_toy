# TVM을 VS Code에서 한 단계씩 따라가기

이 프로젝트는 **4×4 행렬곱 뒤 ReLU**를 세 번 구현합니다. 동일한 입력과 결과를 유지하면서, TVM이 계산을 어떤 표현으로 보관하고 어떻게 CPU 실행 코드로 바꾸는지 살펴봅니다. GPU, 모델 다운로드, 학습 프레임워크는 필요하지 않습니다.

```text
순수 Python 반복문         → 계산 자체 이해
Relax 그래프              → 무엇을 계산할지 표현
Relax pass                → 연산을 TensorIR로 내리고 결합
TensorIR + Schedule       → 버퍼, 루프, 누산과 실행 순서 표현
LLVM 대상 컴파일          → CPU용 코드 생성
Relax VM / TVM 런타임     → 컴파일된 함수를 입력 텐서로 실행
```

## 1. 실행 환경

이 저장소에는 Linux/WSL용 VS Code 실행 설정이 있습니다. 이 컴퓨터의 `.venv`에는 Apache TVM 0.27.0.post1과 NumPy가 설치되어 있습니다. 다른 컴퓨터에서는 Python 3.10 이상으로 다음 명령을 실행하세요. TVM wheel에 LLVM 지원이 있어야 CPU 예제가 돌아갑니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import tvm; print(tvm.__version__, tvm.support.libinfo().get("USE_LLVM"))'
```

마지막 명령에서 `0.27.0.post1 ON`을 확인합니다. VS Code는 WSL 안에서 `code .`로 이 폴더를 열고 **Python** 및 **Python Debugger** 확장을 설치하세요. 상태 표시줄에서 `.venv/bin/python`을 인터프리터로 선택합니다. Windows Python에서 열면 제공된 Linux 가상환경을 사용할 수 없습니다.

## 2. 따라가는 방법

VS Code의 **Run and Debug**에서 아래 실행 구성을 차례로 선택해 `F5`를 누릅니다. 각 파일의 `BREAKPOINT` 주석 바로 다음 실행 줄에 중단점을 찍고 `F10`으로 한 줄씩 진행하세요. `F11`은 호출한 Python 함수 안으로 들어갑니다. Debug Console에서 표의 식을 평가하거나 Watch에 등록할 수 있습니다.

| 순서 | 실행 구성 / 파일 | 멈출 곳 | 볼 것 |
| --- | --- | --- | --- |
| 1 | `1. 순수 Python 계산` / `examples/01_reference.py` | BREAKPOINT 1, 2 | `i`, `j`, `k`, `acc`, `a[i,k]`, `b[k,j]`; 한 출력 원소가 네 번 누산되는 과정 |
| 2 | `2. TensorIR와 스케줄` / `examples/02_tensorir.py` | BREAKPOINT 3, 4 | `MatmulRelu.script()`, `schedule.mod.script()`; `j` 루프가 `j_0 × j_1`로 분리됨 |
| 3 | 같은 파일 | BREAKPOINT 5, 6 | `lib.kind`, `a_tvm`, `c_tvm.numpy()`; IR 컴파일 뒤 런타임 호출 |
| 4 | `3. Relax와 VM` / `examples/03_relax.py` | BREAKPOINT 7, 8 | `Graph.script()`, `info.name`, `mod.script()`; pass마다 바뀌는 IR |
| 5 | 같은 파일 | BREAKPOINT 9, 10, 11 | `lowered.script()`, `vm`, `result`; Relax가 VM과 컴파일된 커널을 연결하는 방식 |

디버거 대신 터미널로 전체 흐름을 먼저 보려면:

```bash
.venv/bin/python examples/01_reference.py
.venv/bin/python examples/02_tensorir.py
.venv/bin/python examples/03_relax.py
```

세 출력은 모두 같습니다.

```text
[[0. 5. 5. 0.]
 [3. 2. 0. 0.]
 [0. 4. 6. 0.]
 [4. 0. 0. 7.]]
```

## 3. 무엇을 비교할까

예제를 실행하면 `artifacts/`에 IR 텍스트가 생성됩니다. VS Code에서 두 파일을 선택해 **Select for Compare** / **Compare with Selected**로 비교하세요.

1. `01_tensorir_original.py` ↔ `02_tensorir_scheduled.py`: `j`가 길이 2인 두 루프로 나뉩니다. `Y[vi,vj] += ...`라는 계산은 같습니다. `03_schedule_trace.txt`에는 이를 재현하는 스케줄 명령이 남습니다.
2. `05_relax_original.py` ↔ `pass_01.py`: `LegalizeOps`가 `R.matmul`, `R.nn.relu`를 별도의 TensorIR `PrimFunc`와 `R.call_tir`로 바꿉니다.
3. `pass_01.py` ↔ `pass_03.py` ↔ `pass_04.py`: `FuseOps`가 두 연산을 그룹으로 묶고, `FuseTIRInner`가 하나의 `fused_matmul_relu` 커널을 만듭니다.
4. `pass_04.py` ↔ `06_relax_lowered.py`: 사용하지 않는 이전 커널이 정리되고, Relax `main`은 융합 커널을 `call_tir`로 호출합니다.

**핵심 구분:** Relax는 텐서 연산의 관계와 입출력 모양을 다루고, TensorIR는 한 커널의 버퍼 접근과 루프를 다룹니다. Schedule은 TensorIR의 실행 형태를 변환합니다. 이 예제의 `split`은 순서를 바꾸지만 결과는 유지합니다. 실제 속도 향상은 이 작은 4×4 입력으로 판단하지 않습니다.

**디버깅 경계:** `@I.ir_module`, `@T.prim_func`, `@R.function` 아래의 코드는 일반 Python 반복 실행이 아니라 TVMScript 파서가 AST를 읽어 IR을 만드는 선언입니다. 따라서 `T.grid` 본문에 중단점을 찍어 CPU의 곱셈을 한 번씩 따라갈 수는 없습니다. 실제 산술은 1단계 Python 예제로, 컴파일 과정은 2·3단계의 Python API 호출과 pass callback에서 관찰하세요. `TVM 내부 Python 추적 (현재 파일)` 구성은 TVM 패키지의 Python 코드에도 `F11`로 들어가게 합니다. C++ 컴파일러와 생성된 기계어 내부 디버깅은 별도의 소스 빌드와 네이티브 디버거가 필요합니다.

## 4. 직접 바꿔 볼 것

- `examples/02_tensorir.py`의 `factors=[None, 2]`를 `[None, 4]`로 바꾸고 IR의 루프 길이를 비교합니다. 결과 행렬은 같아야 합니다.
- `examples/common.py`의 입력값 하나를 음수로 바꾸고, 세 단계에서 바뀐 출력 원소를 찾아갑니다.
- `examples/03_relax.py`에서 `PassRecorder.run_after_pass`에 중단점을 걸고 `LegalizeOps`, `FuseOps`, `FuseTIRInner` 이후 `mod.script()`를 차례로 확인합니다.

## 공식 자료

- [Apache TVM 설치](https://tvm.apache.org/docs/install/index.html)
- [TVM Quick Start: 모델 → 최적화 → VM 실행](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html)
- [TensorIR 생성](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
- [TensorIR 스케줄 변환](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html)
- [Relax VM 구조](https://tvm.apache.org/docs/arch/relax_vm.html)

이 프로젝트는 `requirements.txt`의 TVM 0.27.0.post1을 대상으로 실제 실행했습니다. 웹의 최신 문서는 배포판보다 앞서 API 표기가 달라질 수 있으므로, 코드의 `@T.prim_func(s_tir=True)`처럼 설치된 버전의 API를 기준으로 살펴보세요.
