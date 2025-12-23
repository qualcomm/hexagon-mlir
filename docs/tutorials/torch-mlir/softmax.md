# 🚀 Softmax Torch-mlir Workflow

> **Overview**: This tutorial shows the compilation of Softmax torch-mlir workflow

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 Softmax PyTorch Test](#-softmax-pytorch-test)
  - [📝 Python Test](#-python-test)
  - [⚙️ Test Configuration](#️-test-configuration)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Torch FX graph](#torch-fx-graph)
  - [Torch FX graph to Linalg](#torch-fx-graph-to-linalg)

---

## 🔄 Compilation Pipeline Overview

The hexagon-mlir workflow for Torch undergoes a multi-stage compilation process.

### 🔧 Compilation Stages

| Stage | Tool/Library | Description |
|-------|-------------|-------------|
| **1** | [torch-mlir-compiler](https://github.com/llvm/torch-mlir) | `PyTorch → Linalg IR` |
| **2** | `linalg-hexagon-opt`/libcall | `Linalg IR → MLIR-LLVM` (ML lowering and opt) |
| **3** | `linalg-hexagon-translate` | `MLIR-LLVM IR → Object Code` |
| **4** | Linker | Create `lib_mlir_ciface_Softmax.so` |
| **5** | Runtime | Load and execute on device |

> **💡 Key Insight**: Stage 2 and 3 is the heart of our compiler, leveraging HMX and HVX engines, TCM (Tightly Coupled Memory), DMA transfers, multi-threading support, and custom ML optimizations.

---

## 🧮 Softmax PyTorch Test

### 📝 Python Test

```python
import torch
import torch.nn as nn
import torch_mlir.fx as fx_mlir
from pathlib import Path
import subprocess
import os
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(x)

def create_linalg_module(model, inputs, func_name):
    return fx_mlir.export_and_import(
        model,
        *inputs,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )

def write_bytecode_to_file(mlir_module, filename):
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)

def execute_and_compare(model, inputs, filename, func_name, rtol=1e-05, atol=1e-08, iterations=1, options=None):
    reference = model(*inputs)
    output = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(filename), inputs, func_name, base_dir_for_artifacts=None, iterations=iterations, options=options
    )

    print("\nReference output:\n", reference)
    print("\nHexagon output:\n", output[0])
    assert torch.allclose(output[0], reference, rtol, atol)

def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data as path to process_lwp.py is unknown")
        return

    subprocess.run(
        [
            "python3",
            f"{HEXAGON_MLIR_ROOT}/test/python/process_lwp.py",
            "/tmp/lwp.json",
            "/tmp/lwp_infodump.txt",
            "/tmp/initial-linalg.mlir"
        ],
        check=True
    )

def test_softmax_torch(enablelwp=False): # Set to True to profile with Light Weight Profiling.
    model = Softmax()
    inp = torch.rand(128, 128)
    func_name = model.__class__.__name__
    linalg_filename = Path(__file__).parent / f"{func_name}.mlirbc"

    options = HexagonOptions().__dict__ if enablelwp else None
    if enablelwp:
        options['enableLWP'] = True

    mlir_module = create_linalg_module(model, [inp], func_name)
    write_bytecode_to_file(mlir_module, linalg_filename)
    execute_and_compare(model, [inp], linalg_filename, func_name, options=options)

    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    test_softmax_torch()
```

### ⚙️ Test Configuration

The test can be configured with the following variables:

- **`enablelwp=False`**: If set to True, it will run Light Weight Profiling Tool for the test.

---

## 🏃 Compilation and Execution

### 🔨 Build Commands

```bash
# Enable MLIR IR and LLVM-IR dumping to see intermediate representations
MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 python test_softmax_torch.py
```

### ✅ Expected Output

If setup and device acquisition work correctly, you should see:

```bash
Test_Info: {
	Name:_mlir_ciface_Softmax
	Result:Pass
}
==> Ran successfully, collecting results
PASSED                              
```

---

## 🔍 Deep Dive into Lowering

This section explores the intermediate representations at each compilation stage.

### Torch FX graph

**Stage 1.1**: Torch-mlir converts the model into a GraphModule called FX graph.
It can be printed from torch-mlir's export_and_import() function.

```mlir
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[128, 128]"):
            # File: <...>:16 in forward, code: x = self.softmax(x)
        softmax: "f32[128, 128]" = torch.ops.aten.softmax.int(x, 1);  x = None
        return (softmax,)
```

### Torch FX graph to Linalg

**Stage 1.2**: After torch-mlir pipeline lowers the GraphModule to linalg IR.

This is the starting point for the hexagon-mlir pipeline.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @Softmax(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<128xi64>) -> tensor<128xi64>
    %2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<128x128xf32>) outs(%3, %1 : tensor<128xf32>, tensor<128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.maximumf %in, %out : f32
      %15 = arith.cmpf ogt, %in, %out : f32
      %16 = arith.select %15, %13, %out_1 : i64
      linalg.yield %14, %16 : f32, i64
    } -> (tensor<128xf32>, tensor<128xi64>)
    %expanded = tensor.expand_shape %4#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.subf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x128xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<128x128xf32>
    %8 = tensor.empty() : tensor<128x1xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<128x128xf32>) outs(%9 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<128x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.divf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<128x128xf32>
    return %11 : tensor<128x128xf32>
  }
}
```


- All the IRs after each passes and files can be found from debug prints and inspected.
- Similar torch-mlir tests can be found at test/python/torch-mlir.


> **🎉 Congratulations!** You've successfully walked through the complete compilation pipeline from high-level Python PyTorch code to optimized Hexagon assembly, witnessing the sophisticated optimizations that make efficient ML computation possible on Hexagon architecture.
