# Gemma runner and device sweeps

This directory contains the scripts used to compile and run the Gemma model on
Hexagon, and to benchmark combinations of compiler optimizations.

The main entry point is `run_gemma.py`. It is a driver around:

1. `compile_xmb.py`, which lowers the Gemma linalg module to a Hexagon object.
2. `test_gemma_from_obj.py`, which links the object, pushes artifacts to device,
   executes, and reports the result and performance.
3. `sweep.py`, which repeats those two phases for every combination in a sweep
   JSON file.

## Prerequisites

All standard requirements of hexagon-mlir standalone launcher execution.

## LLVM code-generation knobs

LLVM backend `cl::opt` values can be passed through Gemma driver and saved in
`gemma_options.json`:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --hexagon-llvm-args=-enable-local-pull-up=false \
  --skip-run
```

The Gemma driver passes no raw LLVM backend flags by default. Use `--hexagon-llvm-args`
to experiment with individual backend settings.

For a compile-time-focused build, `--quick-compile` bundles these LLVM
settings: `-enable-local-pull-up=false`, `-hexagon-insert=false`,
`-hexagon-bit=false`, `-rdf-opt=false`, and
`-disable-hexagon-amodeopt`. It is disabled by default as its effects on execution
performancer and correctness are not well-studied.

Gemma dependencies are also required. Download the model checkpoint directories
and pass them via `--checkpoint-path`.

The Gemma root normally contains:

```text
<gemma-root>/
  gemma.linalg
  weights.safetensors
  weights_metadata.json
  tokenizer/
```

## Standard run

`--gemma-root` is the only required argument for a standard end-to-end run.
For example:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT
```

Unless overridden, the driver derives the model paths from `--gemma-root`:

| Item | Default |
| --- | --- |
| Linalg module | `<gemma-root>/gemma.linalg` |
| Weights | `<gemma-root>/weights.safetensors` |
| Weight metadata | `<gemma-root>/weights_metadata.json` |
| Tokenizer | `<gemma-root>/tokenizer` |
| Output directory | `/tmp/gemma_<YYYYMMDD_HHMMSS>` |
| Object file | `<out-dir>/gemma_kernel.o` |


The driver writes the shared options used by compilation and execution to
`<out-dir>/gemma_options.json`. Options can be inspected with:

```bash
python scripts/gemma_specific_scripts/run_gemma.py --help
```

## Debug features

Debug options are intended for individual runs. They are rejected when supplied to a
parameter sweep run.

### Compile only and reuse an object

Compile without running on a device:

```bash
export GEMMA_OUT="/tmp/gemma_$(date +%Y%m%d_%H%M%S)"

python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --out-dir "$GEMMA_OUT" \
  --skip-run
```

This leaves the compiled object in `$GEMMA_OUT`. To run an existing object, use the
same output directory and options that were used to compile it, and add `--skip-compile`:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --out-dir "$GEMMA_OUT" \
  --obj "$GEMMA_OUT/gemma_kernel.o" \
  --skip-compile
```

The object and the `gemma_options.json` supplied to the runtime should describe
the same compilation. Reusing an object with incompatible options can produce
invalid results or link/runtime failures.

### Chop a compiled module

`--chop-after-call` asks the compiler to produce an object that stops after a
selected outlined call. `--chop-occurrence` selects which occurrence to use
when the call name appears more than once.

First compile the chopped object:

```bash
export GEMMA_CHOP="/tmp/gemma_chop_$(date +%Y%m%d_%H%M%S)"

python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --out-dir $GEMMA_CHOP \
  --chop-after-call <outlined-call-name> \
  --chop-occurrence 1 \
  --skip-run
```

Then run the chopped object in `--chop-debug` mode:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --chop-debug \
  --out-dir $GEMMA_CHOP \
  --obj $GEMMA_CHOP/gemma_kernel.o \
  --skip-compile
```

In this mode the driver does not require the full Gemma root or model inputs.
Compilation writes a `.return_type` sidecar next to the object. The debug run
uses that sidecar to generate a small synthetic linalg stub containing the
chopped function's return type.

## Parameter sweeps

A sweep benchmarks every Cartesian-product combination in a JSON file. Each
combination is compiled and then executed as a separate run.

Sweep JSON keys are `GemmaOptions` field names. Only optimization flags are accepted.
A JSON value may be a list or scalar; a scalar is treated as a one-element list.

For example, this file runs the four VTCM-tiling/double-buffering combinations:

```json
{
  "enable_vtcm_tiling": [true, false],
  "enable_double_buffering": [true, false]
}
```

The default sweep config is:

```text
$HEXAGON_MLIR_ROOT/scripts/gemma_specific_scripts/sweep_options.json
```

Run it with `--sweep`:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --sweep
```

Use `--sweep-json` for a custom file. `--sweep` and `--sweep-json` are
mutually exclusive:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --sweep-json /tmp/vtcm_double_buffering.json \
  --sweep-output-dir /tmp/gemma_sweep
```

### Device-pool sweeps

Without `--device-pool`, sweep runs execute sequentially on the configured
device. With `--device-pool`, the driver rents devices and runs combinations
concurrently:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --sweep-json /tmp/vtcm_double_buffering.json \
  --device-pool \
  --num-devices 2 \
  --sweep-output-dir /tmp/gemma_vtcm_db_pool
```

The default device-pool settings are:

| Argument | Default |
| --- | --- |
| `--device-pool` | disabled |
| `--num-devices` | 10 |
| `--timeout` | 120 minutes |
| `--codename` | `kaanapali` |
| `--drental` | `docker run --rm docker-registry.qualcomm.com/hexnn_infra/device-rental:default` |

For each pool slot, the driver runs the following to acquire the device:

```text
<drental> acquire --device-timeout <timeout> -r codename=<codename>
```

### Sweep outputs

The default sweep output directory is:

```text
/tmp/gemma_sweep_<YYYYMMDD_HHMMSS>/
```

The default CSV is `sweep_results.csv` inside that directory. Override either
with `--sweep-output-dir` or `--sweep-csv`.

A sweep directory contains one subdirectory per combination:

```text
gemma_sweep_20260918_094553/
  sweep_results.csv
  device_pool.log                 # only with --device-pool
  0000/
    compile.log
    compile_stdout.log
    execute.log
    gemma_kernel.o
    gemma_options.json
    logcat.txt
    ...
  0001/
    ...
```

Run directories use only their zero-padded combination index. The complete
combination is recorded in `gemma_options.json` and in the corresponding CSV
row.

The CSV has one row per combination and only these result columns:

```text
<one column per JSON field>,compile_time_s,perf_execution_time_s,error
```

- `compile_time_s` is the backend compilation time parsed from the
  `Done in ...s` compiler message.
- `perf_execution_time_s` is the execution performance reported by the device
  run, converted to seconds.

For example, a successful four-combination run has the following shape:

```csv
enable_vtcm_tiling,enable_double_buffering,compile_time_s,perf_execution_time_s,error
true,true,510.3,222.372313,
true,false,520.1,289.640117,
false,true,461.7,685.058801,
false,false,460.0,686.223089,
```

If one combination fails, the sweep continues with the remaining combinations.
Inspect that combination's `compile.log` or `execute.log`. For a device-pool
failure also inspect `device_pool.log`.

## Performance recording

`--record-perf` and `--record-perf-dir <dir>` save timing data from the
measured model invocation (not the warm-up) in two CSVs:

- `<model>_runs.csv` or `<model>_run_<timestamp>.csv` has one run-summary
  row. `run_id` and `timestamp` identify the run; `perf_total_us` is the
  wrapper's `Perf` time; `main_dt_us` is the DSP root-function time; and
  `entry_symbol` is that root function. `commit`, `branch`, `dirty`, and
  `patch_path` record Git state. `kernels_csv` links to the kernel CSV;
  `options_json`, `linalg`, `obj`, `out_dir`, `logcat`, and `prompts` record
  the run inputs and settings. `comment` is an optional string passed with
  `--record-perf-comment` to add a tag or brief note for that run, such as
  `baseline` or `after-tiling`.
- `<model>_kernels_<timestamp>.csv` has one unique instrumented function
  invocation per row, including the root function. The root row's `duration_us` equals
  `main_dt_us`, so its `model_time_pct(x_t/main_t)*100` value is `100%`: for
  this row, `x_t` and `main_t` are both the `main_dt_us` value.
  `seq` is DSP call-start order, reconstructed from the
  `HAP_perf_get_time_us()` timestamps after duplicate logcat records are
  removed. It is not log-delivery order; records from different DSP threads
  can begin at the same time. `symbol` is the function name; `kernel_id`,
  `parent_id`, `parent_symbol`, and `depth` describe its recorded call
  relationship. `duration_us` is its inclusive DSP duration;
  `model_time_pct(x_t/main_t)*100` is the percentage of
  root-function time, where `x_t` is that row's `duration_us` and `main_t` is
  `main_dt_us`;
  `func_before_glue` names the prior invocation on the same DSP thread; and
  `gluecode_exec_time_us` is the time between that invocation's end and this
  invocation's start.

Use either recording mode on an individual device run. `--record-perf` keeps
a shared history; `--record-perf-dir <dir>` keeps both files with this run and
does not update the shared history.

Shared-history mode:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --out-dir <dir> \
  --record-perf \
  --record-perf-comment "baseline"
```

This appends one row to
`SHARED_SUMMARY_DIR/<model>_runs.csv`, where `SHARED_SUMMARY_DIR` is the
shared record path configured in `record_perf.py`:
the directory set by `GEMMA_PERF_SUMMARY_DIR` (defaults to `record_snapshot_perf` in the current directory).
It writes `<model>_kernels_<timestamp>.csv` to `--out-dir`.

Appending checks that the existing file's header matches the current summary
columns. If it does not, because the file was written before a column was added,
removed, or reordered, the recorder reports the mismatch and appends nothing
rather than filing values under the wrong column names. Move the old file aside
or migrate it to resume recording.

Per-run-directory mode:

```bash
python scripts/gemma_specific_scripts/run_gemma.py \
  --gemma-root $GEMMA_ROOT \
  --out-dir <dir> \
  --record-perf-dir <dir>
```

`--record-perf-dir` enables recording by itself. It writes both CSVs to its
directory and does not append to the shared path:

```text
<record-perf-dir>/
  <model>_run_<timestamp>.csv
  <model>_kernels_<timestamp>.csv
```

`--record-perf-comment <text>` stores an annotation in the summary row in
either mode.

`<model>` comes from `--gemma-root`. For a packaged model path ending in
`/deps`, it uses the parent directory name: `.../<model>/deps` produces
`<model>_runs.csv`.

The wrapper runs a warm-up followed by the measured call. The recorder finds
the root `[kernel-time]` record with the latest DSP end timestamp, then keeps
the unique records within that root's DSP time interval. This avoids
mistaking reordered or repeated logcat messages for a new invocation.
`parent=0` identifies an outermost instrumented frame on its DSP thread. The
controller model call normally has the latest end timestamp. If it and a
worker frame end in the same microsecond, the recorder chooses the longer
model call because it includes waiting for the worker work.
Performance recording requires a device run and is not supported with sweeps.
When using `--skip-compile`, reuse an object built with timing instrumentation
so logcat contains `[kernel-time]` records.

### Instrumentation overhead

Every `run_gemma.py` build is instrumented: `compile_xmb.py` sets
`injectProgressPrints` unconditionally, so this applies to plain runs and sweeps
as well as to `--record-perf`, which only decides whether the CSVs are written.
Each selected function adds two DSP timer calls and one return-side `printf`.

The added timer and `printf` code changes total runtime. A function's own
duration excludes the timing code added directly to that function, but a
caller includes timing code added to functions it calls. Therefore,
`gluecode_exec_time_us` is not the total timing-code overhead. Compare
durations only between builds instrumented in the same way; do not compare
their `Perf` values directly with an uninstrumented baseline.
