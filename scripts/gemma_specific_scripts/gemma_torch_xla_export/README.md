# Torch to StableHLO to Hexagon-MLIR Workflow

This workflow:
1. Exports a Hugging Face model checkpoint through Torch-XLA to StableHLO
2. Lowers StableHLO to Linalg via jaxlib passes
3. Packs model weights into a safetensors bundle
4. Compiles the linalg module to an object file with hexagon-mlir
5. Executes the object file on your Qualcomm Hexagon NPU

```text
checkpoint -> torch.export -> StableHLO -> Linalg -> Hexagon object -> device
```

This README covers the `gemma3_300M` selection: an embedding model (not a
causal language model) whose export produces a 768-dimensional embedding after
masked mean pooling, two dense projection layers, and L2 normalization.

## 1. Python Environment

This workflow requires Python 3.12 or newer.

```bash
export REPO_ROOT=/path/to/your/hexagon-mlir
export WORK_DIR="$REPO_ROOT/work"
mkdir -p "$WORK_DIR"

python --version # Expect Python 3.12 or newer.
```

If importing `torch_xla` reports that `libpython3.12.so` cannot be found, add
the active Python installation's `lib` directory to `LD_LIBRARY_PATH`.

## 2. Model Dependencies

```bash
pip install -r "$REPO_ROOT/ci/hexagon-mlir-requirements.txt"
pip install -r "$REPO_ROOT/ci/stablehlo-requirements-pub.txt"
```

## 3. Build Hexagon-MLIR

Follow the instructions in [docs/user-guide.md](../../../docs/user-guide.md) from
**System Packages** through the **Script-based Build** section, up to and including:

```bash
source scripts/set_local_env.sh
bash ./scripts/build_hexagon_mlir.sh
```

Set the environment variables for your local device.

```bash
export ANDROID_HOST="<ANDROID_HOST returned by acquire>"
export ANDROID_SERIAL="<ANDROID_SERIAL returned by acquire>"
```

## 4. Gemma3 300M Embedding Model

Download the Gemma3 300M checkpoint via the Hugging Face API to `$REPO_ROOT/gemma3_300M`

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf download google/embeddinggemma-300m --local-dir $REPO_ROOT/gemma3_300M
```

The example command uses a batch of two so the device test can compare two embeddings.
The prompts passed to the device runner include the checkpoint's query prefix; the exporter reads that
prefix from `config_sentence_transformers.json` itself.

```bash
export GEMMA3_CHECKPOINT="$REPO_ROOT/gemma3_300M"
export GEMMA3_OUT="$WORK_DIR/output_gemma3_300M"
export GEMMA3_RUN="$WORK_DIR/run_gemma3_300M"
export GEMMA3_PROMPTS='task: search result | query: What is the capital of France?,task: search result | query: What is the capital of Italy?'

python \
  "$REPO_ROOT/scripts/gemma_specific_scripts/gemma_torch_xla_export/driver.py" \
  --model gemma3_300M \
  --checkpoint-path "$GEMMA3_CHECKPOINT" \
  --output-dir "$GEMMA3_OUT" \
  --mode prefill \
  --no-kv-cache \
  --context-length 128 \
  --batch-size 2 \
  --prompt "What is the capital of France?"

python "$REPO_ROOT/scripts/gemma_specific_scripts/gemma_torch_xla_export/make_weights_safetensors.py" \
  --data-dir "$GEMMA3_OUT/data" \
  --forward-meta "$GEMMA3_OUT/functions/forward.meta" \
  --weights "$GEMMA3_OUT/weights.safetensors" \
  --weights-metadata "$GEMMA3_OUT/weights_metadata.json"

python "$REPO_ROOT/scripts/gemma_specific_scripts/run_gemma.py" \
  --linalg "$GEMMA3_OUT/linalg.mlir" \
  --weights "$GEMMA3_OUT/weights.safetensors" \
  --metadata "$GEMMA3_OUT/weights_metadata.json" \
  --tokenizer-dir "$GEMMA3_CHECKPOINT" \
  --prompts "$GEMMA3_PROMPTS" \
  --simple-extend-va \
  --out-dir "$GEMMA3_RUN"
```

The device test detects the `[batch, 768]` Linalg result as "embedding mode",
tokenizes the two prompts, runs the graph on device, and reports the cosine
similarity. Keep `--batch-size` equal to the number of comma-separated prompts.
For a single embedding, use batch size 1 and one prompt.

### With HexKL and Multithreading

To improve performance, enable HexKL kernels, multithreading, and FP32-to-FP16
relaxation. FP32 matmul relaxation lowers FP32 matrix multiplications to FP16,
which HexKL can accelerate at the cost of reduced numerical precision.

```bash
python "$REPO_ROOT/scripts/gemma_specific_scripts/run_gemma.py" \
  --linalg "$GEMMA3_OUT/linalg.mlir" \
  --weights "$GEMMA3_OUT/weights.safetensors" \
  --metadata "$GEMMA3_OUT/weights_metadata.json" \
  --tokenizer-dir "$GEMMA3_CHECKPOINT" \
  --prompts "$GEMMA3_PROMPTS" \
  --simple-extend-va \
  --out-dir "$GEMMA3_RUN" \
  --enable-hexkl \
  --relax-fp32-matmul-to-fp16 \
  --enable-multi-threading
```

## 5. How HVX and HMX Are Used

The compiler maps every op in the model to one of three execution paths:
HVX vector instructions (most ops), HVX runtime library calls (transcendentals),
or the HMX matrix-multiply engine (matmuls). The IR dump at
`compile_irdump.mlir` (produced by passing `--dump-ir` to `run_gemma.py`)
shows each of these paths in the same file.

### HVX: element-wise ops map 1-to-1 to vector instructions

Arithmetic ops on HVX-width vectors (32 × f32 = 128 bytes = one HVX register)
lower directly to a single HVX instruction via LLVM's Hexagon back-end.  No
library call, no loop — one MLIR op becomes one assembly instruction:

```mlir
// After vectorization (from compile_irdump.mlir):
%15 = arith.addf %9, %14  : vector<64xf32>   // → V6_vadd_sf  (two HVX regs)
%16 = arith.mulf %7, %11  : vector<64xf32>   // → V6_vmpy_sf
%17 = arith.mulf %16, %15 : vector<64xf32>   // → V6_vmpy_sf
%18 = arith.truncf %17    : vector<64xf32> to vector<64xf16>  // → V6_vpackl
```

In LLVM IR the same values appear as `llvm.fadd`, `llvm.fmul`, etc. on
`vector<32xf32>` (one physical HVX register); LLVM then selects the
corresponding `V6_vadd_sf` / `V6_vmpy_sf` / `V6_vmpyhss` instructions.
Loads and stores are similarly direct:

```mlir
%34 = llvm.load %ptr {alignment = 4} : !llvm.ptr -> vector<64xf32>
      llvm.store %result, %out {alignment = 4} : vector<64xf32>, !llvm.ptr
```

### HVX runtime libraries: transcendentals

A handful of ops have no single HVX instruction.  `math.exp`, `math.tanh`,
`math.rsqrt`, `math.sqrt`, `llvm.intr.pow`, `llvm.intr.cos`, and
`llvm.intr.sin` are lowered to calls into HVX-vectorized runtime libraries
(`_hexagon_runtime_*__vf` or `qhmath_*`).  These libraries are written with
HVX intrinsics directly and process a full vector per call.

Softmax attention in Gemma (from `compile_irdump.mlir`):

```mlir
// math.exp on tensor<2x3x128x128xf32>:
%mapped_139 = linalg.map { math.exp }
    ins(%mapped_138 : tensor<2x3x128x128xf32>)
   outs(%95        : tensor<2x3x128x128xf32>)
```

After full lowering this becomes:

```mlir
// LLVM IR — vectorized loop body, 32 floats per call:
%49 = llvm.fsub %43, %48 {fastmathFlags = #llvm.fastmath<reassoc>} : vector<32xf32>
%50 = llvm.bitcast %49 : vector<32xf32> to vector<32xi32>
%51 = llvm.call @_hexagon_runtime_exp__vf(%50) : (vector<32xi32>) -> vector<32xi32>
%52 = llvm.bitcast %51 : vector<32xi32> to vector<32xf32>
```

GeLU activation uses `math.tanh` (from the same dump):

```mlir
%mapped_203 = linalg.map { math.tanh }
    ins(%mapped_202 : tensor<2x128x1152xf32>)
   outs(%145        : tensor<2x128x1152xf32>)
```

Which lowers to:

```mlir
%59 = llvm.call @_hexagon_runtime_tanh__vf(%58) : (vector<32xi32>) -> vector<32xi32>
```

RMS-norm uses `math.rsqrt`; scalar fallback uses `qhmath_rsqrt_f`:

```mlir
%mapped_28 = linalg.map { math.rsqrt }
    ins(%mapped_27 : tensor<2x128x1xf32>)
   outs(%10        : tensor<2x128x1xf32>)
// ... lowers to:
%35 = llvm.call @qhmath_rsqrt_f(%34) : (f32) -> f32
```

The `_hexagon_runtime_*__vf` functions are part of the hexagon runtime (see
`qcom_hexagon_backend/bin/runtime/`); `qhmath_*` functions come from the
Hexagon SDK's scalar math library.

### HMX: matmuls via the HexKL micro API

The model's projection layers are `linalg.matmul` ops.  When
`--enable-hexkl --relax-fp32-matmul-to-fp16` is active, the compiler first
relaxes FP32 inputs to FP16, then converts each `linalg.matmul` to a single
`hexkl.matmul` op:

```mlir
// After RelaxFP32MatmulToFP16 + MatmulToHexKL (from compile_irdump.mlir):
%13 = hexkl.matmul
    ins(%9, %11 : tensor<256x768xf16>, tensor<768x768xf16>)
   outs(%12     : tensor<256x768xf32>) -> tensor<256x768xf32>

%32 = hexkl.matmul
    ins(%28, %30 : tensor<256x768xf16>, tensor<768x256xf16>)
   outs(%31      : tensor<256x256xf32>) -> tensor<256x256xf32>
```

`DecomposeHexKLMatmulPass` then tiles each `hexkl.matmul` into a tiled loop
nest that calls the HMX micro API — a low-level API that programs the HMX
accumulator directly:

```mlir
// Tiled LLVM IR emitted by DecomposeHexKLMatmulPass (compile_irdump.mlir):

// 1. Copy activation tile to VTCM and convert to AH (activation-halved) layout
llvm.call @hexkl_micro_hmx_copy_submatrix_to_f16(...)
llvm.call @hexkl_micro_hmx_rm_to_ah_f16(%act_vtcm, %act_offset, %k_offset)

// 2. Per K-tile: clear accumulator, load weight tile to WH layout, run mm
llvm.call @hexkl_micro_hmx_acc_clear_f16()
llvm.call @hexkl_micro_hmx_rm_to_wh_f16(%wgt_vtcm, %c98304_i32, %wgt_src,
                                          %wgt_offset, %k_offset, %c768_i32)
llvm.call @hexkl_micro_hmx_mm_f16(%acc_ptr, %tile_offset, %c98304_i32)

// 3. After K-loop: read accumulator and convert result back to row-major FP32
llvm.call @hexkl_micro_hmx_acc_read_f16(%acc_buf, %read_offset, %c102400_i32)
llvm.call @hexkl_micro_hmx_ah_to_rm_f16(%out_vtcm, %c98304_i32, %c102400_i32)
```

The sequence is: load activations → load weights → `acc_clear` → `mm` (one
K-tile per call) → `acc_read` → write result.  Every step runs on the HMX
engine inside the DSP core.

### Summary

| Op class | IR example | Lowering |
|---|---|---|
| `arith.addf`, `arith.mulf` on `vector<32xf32>` | `%15 = arith.addf %9, %14 : vector<64xf32>` | Single HVX instruction (`V6_vadd_sf`) |
| `math.exp`, `math.tanh` on tensors | `linalg.map { math.exp } ins(tensor<2x3x128x128xf32>)` | `@_hexagon_runtime_exp__vf` / `@_hexagon_runtime_tanh__vf` (HVX intrinsic library) |
| `math.rsqrt` scalar residual | `linalg.map { math.rsqrt } ins(tensor<2x128x1xf32>)` | `@qhmath_rsqrt_f` (Hexagon SDK scalar math) |
| `linalg.matmul` with `--enable-hexkl` | `hexkl.matmul ins(tensor<256x768xf16>, tensor<768x768xf16>)` | HMX micro API: `acc_clear` → `rm_to_wh` → `mm` → `acc_read` |

## 6. Outputs

The export directory contains `stablehlo.mlir`, `linalg.mlir`,
`functions/forward.meta`, exported parameter arrays under `data/`, and constants
under `constants/`. The weight-packing step creates `weights.safetensors` and
`weights_metadata.json`; `run_gemma.py` writes the object, logs, and execution
artifacts to the selected run directory.
