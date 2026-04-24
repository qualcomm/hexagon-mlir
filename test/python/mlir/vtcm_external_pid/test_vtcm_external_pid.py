# ===- test_vtcm_external_pid.py --------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Tests external VTCM scratch with pid-indexed offsets.
#
# The kernel receives the full VTCM buffer and uses pid_x to select its own
# private scratch region, mirroring how it partitions input/output data.
#
# Pre-processing: input.mlir is run through linalg-hexagon-opt -memory-offsets
# before being passed to the LinalgToLLVM pipeline. This simulates what will
# happen once MemoryOffsetsPass is wired into the pipeline via vtcmExternal.
#
# ===------------------------------------------------------------------------===

import pytest
import os
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triton.backends.qcom_hexagon_backend.mlir_launcher import MLIRHexagonLauncher
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


@pytest.mark.parametrize("grid", [1, 2, 4])
def test_vtcm_external_pid(grid):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_name = "vtcm_external_pid"

    # Calculate grid-dependent sizes
    tile_size = 64
    n_elements = grid * tile_size
    per_instance_vtcm = 4096  # Fixed chunk size passed by wrapper

    # Dynamically generate the MLIR template
    mlir_content = textwrap.dedent(
        f"""\
        func.func @vtcm_external_pid(
            %A: memref<{n_elements}xf32>,
            %B: memref<{n_elements}xf32>,
            %C: memref<{n_elements}xf32>,
            %vtcm: memref<{per_instance_vtcm}xi8, 1> {{hexagon.scratch}},
            %num_prog_x: i32, %num_prog_y: i32, %num_prog_z: i32,
            %pid_x: i32, %pid_y: i32, %pid_z: i32) {{

          %c64  = arith.constant 64  : index
          %pid  = arith.index_cast %pid_x : i32 to index

          // Each instance selects its 64-element tile (data partitioning by pid).
          %tile_offset = arith.muli %pid, %c64 : index
          %tile_A = memref.subview %A[%tile_offset] [64] [1]
              : memref<{n_elements}xf32> to memref<64xf32, strided<[1], offset: ?>>
          %tile_B = memref.subview %B[%tile_offset] [64] [1]
              : memref<{n_elements}xf32> to memref<64xf32, strided<[1], offset: ?>>
          %tile_C = memref.subview %C[%tile_offset] [64] [1]
              : memref<{n_elements}xf32> to memref<64xf32, strided<[1], offset: ?>>

          // Two VTCM scratch allocs — MemoryOffsetsPass will replace these with
          // pid-indexed views into %vtcm.
          %scratch_A = memref.alloc() : memref<64xf32, 1>
          %scratch_B = memref.alloc() : memref<64xf32, 1>

          // Stage inputs to VTCM.
          memref.copy %tile_A, %scratch_A
              : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 1>
          memref.copy %tile_B, %scratch_B
              : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 1>

          // Add in VTCM: scratch_A = scratch_A + scratch_B.
          linalg.add ins(%scratch_A, %scratch_B : memref<64xf32, 1>, memref<64xf32, 1>)
                     outs(%scratch_A : memref<64xf32, 1>)

          // Write result back to DDR.
          memref.copy %scratch_A, %tile_C
              : memref<64xf32, 1> to memref<64xf32, strided<[1], offset: ?>>

          memref.dealloc %scratch_A : memref<64xf32, 1>
          memref.dealloc %scratch_B : memref<64xf32, 1>
          return
        }}
    """
    )

    # Dynamically generate the C++ wrapper
    cpp_content = textwrap.dedent(
        f"""\
        #include <cmath>
        #include <random>
        #define FARF_ALWAYS 1
        #include "CRunnerUtils.cpp"
        #include "common.h"
        #include "hexagon_benchmark.h"
        #include "prof_utils.h"
        #include "tensor.h"
        #include "test_report.h"
        #include <HAP_compute_res.h>
        #include <HAP_farf.h>
        #include <cstdint>
        #include <stdlib.h>

        const int N = {n_elements};
        const int TILE = 64;
        const int NUM_INSTANCES = {grid};

        const int PER_INSTANCE_VTCM = 4096;
        const int TOTAL_VTCM = NUM_INSTANCES * PER_INSTANCE_VTCM;

        extern "C" void vtcm_external_pid(
            float *, float *, int64_t, int64_t, int64_t,  // A
            float *, float *, int64_t, int64_t, int64_t,  // B
            float *, float *, int64_t, int64_t, int64_t,  // C
            int8_t *, int8_t *, int64_t, int64_t, int64_t, // vtcm scratch
            int, int, int,                                  // num_prog_x/y/z
            int, int, int);                                 // pid_x/y/z

        bool check(float *A, float *B, float *C) {{
          for (int i = 0; i < N; i++) {{
            float expected = A[i] + B[i];
            float diff = expected - C[i];
            if (diff > 1e-3f || diff < -1e-3f) {{
              FARF(ALWAYS, "Mismatch at index %d: %f + %f = %f (got %f, diff %f)\\n", i,
                   A[i], B[i], expected, C[i], diff);
              return false;
            }}
          }}
          return true;
        }}

        int main() {{
          Tensor<float, 1> TA({{{{N}}, {{1}}, 128}}, MemType::HEAP);
          Tensor<float, 1> TB({{{{N}}, {{1}}, 128}}, MemType::HEAP);
          Tensor<float, 1> TC({{{{N}}, {{1}}, 128}}, MemType::HEAP);

          MemRefDescriptor<float, 1> *dA = TA.toMemRefDesc();
          MemRefDescriptor<float, 1> *dB = TB.toMemRefDesc();
          MemRefDescriptor<float, 1> *dC = TC.toMemRefDesc();

          for (int i = 0; i < N; i++) {{
            dA->allocated[i] = static_cast<float>(i);
            dB->allocated[i] = static_cast<float>(i * 2);
            dC->allocated[i] = 0.0f;
          }}

          compute_res_attr_t vtcmRes;
          HAP_compute_res_attr_init(&vtcmRes);
          HAP_compute_res_attr_set_vtcm_param(&vtcmRes, TOTAL_VTCM,
                                              ENFORCE_SINGLE_PAGE_VTCM_ALLOC);
          uint32_t vtcmContextID = HAP_compute_res_acquire(&vtcmRes, HAP_REQUEST_TIMEOUT_US);
          if (vtcmContextID == 0) {{
            FARF(ALWAYS, "Failed to acquire VTCM (bytes=%d)\\n", TOTAL_VTCM);
            return -1;
          }}
          int8_t *vtcm_raw =
              static_cast<int8_t *>(HAP_compute_res_attr_get_vtcm_ptr(&vtcmRes));
          if (vtcm_raw == nullptr) {{
            FARF(ALWAYS, "VTCM acquire returned null pointer\\n");
            HAP_compute_res_release(vtcmContextID);
            return -1;
          }}

          uint64_t avg_cycles = benchmark_time_us(1, [&]() {{
            for (int pid = 0; pid < NUM_INSTANCES; pid++) {{
              int8_t *vtcm_instance_ptr = vtcm_raw + (pid * PER_INSTANCE_VTCM);
              vtcm_external_pid(
                  dA->allocated, dA->aligned, dA->offset, dA->sizes[0], dA->strides[0],
                  dB->allocated, dB->aligned, dB->offset, dB->sizes[0], dB->strides[0],
                  dC->allocated, dC->aligned, dC->offset, dC->sizes[0], dC->strides[0],
                  vtcm_instance_ptr, vtcm_instance_ptr, 0, PER_INSTANCE_VTCM, 1,
                  NUM_INSTANCES, 1, 1,
                  pid, 0, 0);
            }}
          }});

          Result result = Result::Pass;
          if (!check(dA->allocated, dB->allocated, dC->allocated))
            result = Result::Fail;

          HAP_compute_res_release(vtcmContextID);

          TestReport tr("vtcm_external_pid", avg_cycles, "microseconds", result);
          tr.save();

          WriteLWPOutput("/data/local/tmp/lwp.json");

          return 0;
        }}
    """
    )

    # Write files to temp and run
    with tempfile.NamedTemporaryFile(
        suffix=".mlir", delete=False, mode="w"
    ) as tmp_mlir, tempfile.NamedTemporaryFile(
        suffix=".cpp", delete=False, mode="w"
    ) as tmp_cpp, tempfile.NamedTemporaryFile(
        suffix=".mlir", delete=False
    ) as tmp_processed:

        input_mlir = tmp_mlir.name
        cpp_wrapper = tmp_cpp.name
        processed_mlir = tmp_processed.name

        tmp_mlir.write(mlir_content)
        tmp_cpp.write(cpp_content)

    try:
        result = subprocess.run(
            ["linalg-hexagon-opt", input_mlir, "-memory-offsets"],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(processed_mlir, "w") as f:
            f.write(result.stdout)

        options = HexagonOptions().__dict__.copy()
        options["enableConvertToHexagonmem"] = False
        options["enableLWP"] = True
        options = {k: str(v) for k, v in options.items()}

        MLIRHexagonLauncher.run_mlir_with_custom_cpp_wrapper(
            processed_mlir, cpp_wrapper, True, options, kernel_name
        )
    finally:
        os.unlink(input_mlir)
        os.unlink(cpp_wrapper)
        os.unlink(processed_mlir)
