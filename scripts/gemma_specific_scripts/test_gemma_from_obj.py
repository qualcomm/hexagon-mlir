# ===- test_gemma_from_obj.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""
Run a pre-compiled Gemma kernel (.o) on Hexagon (the --from-obj flow).

This is the only flow driven by run_gemma.py (its run_on_device step invokes
this module directly with --from-obj). All paths come from pytest flags (set by
the driver); a few knobs are read from the environment:

  GEMMA4_OBJ                .o to run (falls back to the --linalg-derived name)
  GEMMA4_LINALG             linalg .mlir the .o was lowered from (return sig only)
  GEMMA4_PROMPTS            comma-separated prompts for embedding mode
  GEMMA4_NO_INPUTS=1        fabricate synthetic inputs on device

Steps: StableHLO launcher builds wrapper -> obj->so link -> (optional) execute
on device -> decode.

Inputs come from one of two real-input sources, or are generated synthetically
on device:
  --inputs-dir DIR   pre-generated _mlir_ciface_main_t{i}.raw + manifest.json
  (default)          tokenizer + weights via _build_inputs() from a prompt
  GEMMA4_NO_INPUTS=1 syntheticInputs=True (wrapper fabricates inputs on device)

The function @main operands are assembled in StableHLO export order from
forward.meta: constants, named parameters, and flattened runtime inputs.
"""

from pathlib import Path
import json
import os
import torch
from gemma_options import GemmaOptions

# Paths are supplied at runtime via pytest flags (--linalg/--weights/--metadata/
# --tokenizer-dir) or the matching GEMMA4_* env vars; there are no built-in
# defaults. They are populated in the test body from those flags.
LINALG_FILE = None
WEIGHTS_FILE = None
METADATA_FILE = None
TOKENIZER_DIR = None

SEQ_LEN = 128  # default; overridden by _infer_seq_shape() once the linalg is known
BATCH_SIZE = 1  # default; overridden by _infer_seq_shape()


def _infer_seq_shape(linalg_path: Path):
    """Read batch_size and seq_len from the first tensor<NxMxi32> in the linalg file."""
    import re as _re

    global SEQ_LEN, BATCH_SIZE
    try:
        text = linalg_path.read_text(errors="replace")
        m = _re.search(r"tensor<(\d+)x(\d+)xi32>", text)
        if m:
            BATCH_SIZE = int(m.group(1))
            SEQ_LEN = int(m.group(2))
    except OSError:
        pass


DECODE_EMBEDDING_ROW_MARKER = "// gemma.decode_embedding_rows = true"


def _uses_decode_embedding_rows() -> bool:
    if LINALG_FILE is None:
        return False
    with LINALG_FILE.open(errors="replace") as linalg_file:
        return DECODE_EMBEDDING_ROW_MARKER in linalg_file.read(256)


def _find_forward_meta() -> Path:
    explicit_path = os.environ.get("GEMMA4_FORWARD_META")
    candidates = [Path(explicit_path)] if explicit_path else []
    if METADATA_FILE is not None:
        candidates.extend(
            (
                METADATA_FILE.parent / "functions" / "forward.meta",
                METADATA_FILE.parent / "forward.meta",
            )
        )
    if LINALG_FILE is not None:
        candidates.extend(
            (
                LINALG_FILE.parent / "functions" / "forward.meta",
                LINALG_FILE.parent.parent / "functions" / "forward.meta",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Cannot find StableHLO forward.meta needed to assemble @main inputs. "
        f"Searched: {searched_paths or '(no candidate paths)'}"
    )


def _load_forward_metadata() -> tuple[dict, Path]:
    forward_meta_path = _find_forward_meta()
    forward_meta = json.loads(forward_meta_path.read_text())
    return forward_meta, forward_meta_path.parent.parent / "constants"


def _main_argument_count(linalg_path: Path) -> int:
    import re

    text = linalg_path.read_text(errors="replace")
    match = re.search(r"(?m)^\s*func\.func\s+(?:public\s+)?@main\s*\(", text)
    if match is None:
        raise ValueError(f"Could not find func.func @main in {linalg_path}")
    depth = 1
    comma_count = 0
    has_argument = False
    for character in text[match.end() :]:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return comma_count + int(has_argument)
        elif depth == 1 and character == ",":
            comma_count += 1
        elif depth == 1 and not character.isspace():
            has_argument = True
    raise ValueError(f"Unterminated func.func @main signature in {linalg_path}")


def _runtime_inputs(
    forward_meta: dict,
    input_ids_1d: list[int],
) -> dict[int, torch.Tensor]:
    input_specs = {}
    for location, signature in zip(
        forward_meta["input_locations"], forward_meta["input_signature"]
    ):
        if location["type_"] == "input_arg":
            position = location["position"]
            if position in input_specs:
                raise ValueError(f"Duplicate input_arg position {position}")
            input_specs[position] = signature
    if not input_specs:
        return {}
    expected_positions = list(range(len(input_specs)))
    if sorted(input_specs) != expected_positions:
        raise ValueError(
            "input_arg positions must be contiguous from zero; found "
            f"{sorted(input_specs)}"
        )
    if not input_ids_1d:
        raise ValueError("Cannot build runtime inputs without token IDs")

    ids_signature = input_specs[0]
    ids_shape = tuple(ids_signature["shape"])
    if not ids_shape:
        raise ValueError(f"Expected input_ids tensor, got {ids_signature}")
    sequence_length = ids_shape[-1]
    sequence = input_ids_1d[-sequence_length:]
    pad_length = sequence_length - len(sequence)
    padded_ids = [0] * pad_length + sequence
    batch_size = 1
    for dimension in ids_shape[:-1]:
        batch_size *= dimension
    ids = torch.tensor(
        padded_ids * batch_size,
        dtype=getattr(torch, ids_signature["dtype"]),
    ).reshape(ids_shape)

    runtime_inputs = {0: ids}
    for position, signature in input_specs.items():
        if position == 0:
            continue
        shape = tuple(signature["shape"])
        dtype = getattr(torch, signature["dtype"])
        if position == 1 and shape == ids_shape:
            row = [0] * pad_length + [1] * len(sequence)
            runtime_inputs[position] = torch.tensor(
                row * batch_size, dtype=dtype
            ).reshape(shape)
        elif position == 2 and shape == ids_shape:
            row = [0] * pad_length + list(range(len(sequence)))
            runtime_inputs[position] = torch.tensor(
                row * batch_size, dtype=dtype
            ).reshape(shape)
        else:
            runtime_inputs[position] = torch.zeros(shape, dtype=dtype)
    return runtime_inputs


def _assemble_program_inputs(
    forward_meta: dict,
    constants_dir: Path,
    state_dict: dict[str, torch.Tensor],
    runtime_inputs: dict[int, torch.Tensor],
    token_id: int | None = None,
) -> list[torch.Tensor]:
    import numpy as np

    row_selected_decode = _uses_decode_embedding_rows()
    row_parameters = {
        "language_model.embed_tokens_per_layer.weight_scale": (262144, 35),
        "language_model.embed_tokens_per_layer.weight": (262144, 8960),
    }
    selected_token_row = None
    if row_selected_decode:
        if token_id is None:
            raise ValueError("Cannot select embedding rows without a token ID")
        selected_token_row = min(int(token_id) & 0xFFFFFFFF, 262143)

    locations = forward_meta["input_locations"]
    signatures = forward_meta["input_signature"]
    if len(locations) != len(signatures):
        raise ValueError(
            "forward.meta input_locations/input_signature length mismatch: "
            f"{len(locations)} vs {len(signatures)}"
        )

    inputs = []
    selected_row_parameters = set()
    for argument_index, (location, signature) in enumerate(zip(locations, signatures)):
        kind = location["type_"]
        if kind == "constant":
            constant_path = constants_dir / str(location["position"])
            if not constant_path.is_file():
                raise FileNotFoundError(
                    f"Missing constant for @main argument {argument_index}: "
                    f"{constant_path}"
                )
            array = np.load(constant_path, allow_pickle=False)
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            tensor = torch.from_numpy(array)
        elif kind == "parameter":
            parameter_name = location["name"]
            if parameter_name not in state_dict:
                raise KeyError(
                    f"Parameter {parameter_name!r} for @main argument "
                    f"{argument_index} is missing from the safetensors file"
                )
            tensor = state_dict[parameter_name]
            expected_table_shape = row_parameters.get(parameter_name)
            if row_selected_decode and expected_table_shape is not None:
                if tuple(tensor.shape) != expected_table_shape:
                    raise ValueError(
                        f"Unexpected shape for {parameter_name}: {tuple(tensor.shape)}"
                    )
                tensor = tensor[selected_token_row].contiguous()
                selected_row_parameters.add(parameter_name)
        elif kind == "input_arg":
            position = location["position"]
            if position not in runtime_inputs:
                raise ValueError(
                    f"No runtime tensor supplied for input_arg position {position}"
                )
            tensor = runtime_inputs[position]
        else:
            raise ValueError(
                f"Unknown input kind {kind!r} at @main argument {argument_index}"
            )

        expected_shape = tuple(signature["shape"])
        expected_dtype = getattr(torch, signature["dtype"])
        if (
            kind == "parameter"
            and row_selected_decode
            and location["name"] in row_parameters
        ):
            expected_shape = (expected_shape[-1],)
        elif kind == "parameter" and not expected_shape and tensor.numel() == 1:
            tensor = tensor.reshape(())
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"@main argument {argument_index} ({kind}, "
                f"{location.get('name') or location.get('position')}): expected "
                f"{expected_shape} {expected_dtype}, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
        inputs.append(tensor.contiguous())

    if row_selected_decode and selected_row_parameters != set(row_parameters):
        raise ValueError(
            "Decode row-input mode did not find both embedding parameters: "
            f"{sorted(selected_row_parameters)}"
        )
    return inputs


def _validate_program_inputs(
    inputs: list[torch.Tensor], forward_meta: dict, linalg_path: Path
) -> None:
    locations = forward_meta["input_locations"]
    signatures = forward_meta["input_signature"]
    linalg_count = _main_argument_count(linalg_path)
    if len(locations) != linalg_count:
        raise ValueError(
            f"forward.meta describes {len(locations)} @main inputs, but "
            f"{linalg_path} has {linalg_count}"
        )
    if len(inputs) != linalg_count:
        raise ValueError(
            f"Prepared {len(inputs)} @main inputs, but Linalg requires "
            f"{linalg_count}; input ABI ordering/contents are inconsistent"
        )
    row_parameters = {
        "language_model.embed_tokens_per_layer.weight_scale",
        "language_model.embed_tokens_per_layer.weight",
    }
    row_selected_decode = _uses_decode_embedding_rows()
    for argument_index, (tensor, location, signature) in enumerate(
        zip(inputs, locations, signatures)
    ):
        expected_shape = tuple(signature["shape"])
        if (
            row_selected_decode
            and location["type_"] == "parameter"
            and location["name"] in row_parameters
        ):
            expected_shape = (expected_shape[-1],)
        expected_dtype = getattr(torch, signature["dtype"])
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"@main argument {argument_index} ({location['type_']}): "
                f"expected {expected_shape} {expected_dtype}, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )


def _build_inputs(input_ids_1d: list[int]) -> list[torch.Tensor]:
    import safetensors.torch

    forward_meta, constants_dir = _load_forward_metadata()
    runtime_inputs = _runtime_inputs(forward_meta, input_ids_1d)
    state_dict = safetensors.torch.load_file(str(WEIGHTS_FILE), device="cpu")
    return _assemble_program_inputs(
        forward_meta,
        constants_dir,
        state_dict,
        runtime_inputs,
        token_id=input_ids_1d[-1] if input_ids_1d else None,
    )


def _build_embedding_inputs(prompts: list[str]) -> list[torch.Tensor]:
    """Build embedding inputs and assemble all @main operands from forward.meta."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    ids = enc["input_ids"].to(torch.int32)  # (n_prompts, SEQ_LEN)
    mask = enc["attention_mask"].to(torch.int32)

    # Pad / truncate the batch dimension to BATCH_SIZE.
    n = ids.shape[0]
    if n < BATCH_SIZE:
        pad = BATCH_SIZE - n
        ids = torch.cat([ids, torch.zeros(pad, SEQ_LEN, dtype=torch.int32)], dim=0)
        mask = torch.cat([mask, torch.zeros(pad, SEQ_LEN, dtype=torch.int32)], dim=0)
    elif n > BATCH_SIZE:
        ids = ids[:BATCH_SIZE]
        mask = mask[:BATCH_SIZE]

    forward_meta, constants_dir = _load_forward_metadata()
    runtime_inputs = {0: ids.contiguous(), 1: mask.contiguous()}
    import safetensors.torch

    state_dict = safetensors.torch.load_file(str(WEIGHTS_FILE), device="cpu")
    return _assemble_program_inputs(
        forward_meta,
        constants_dir,
        state_dict,
        runtime_inputs,
    )


def _infer_return_shape(linalg_path: Path):
    """Parse the @main return tensor shape from the linalg signature.

    Returns a list of ints (e.g. [2, 768]) or None if it can't be found. Used to
    auto-detect the run mode: a rank-2 tensor<Bx768xf32> => embedding model,
    otherwise the LLM logits path (tensor<1x128x262144xf32>).
    """
    import re as _re

    try:
        text = linalg_path.read_text(errors="replace")
    except OSError:
        return None
    # The return clause of @main: "... -> (tensor<2x768xf32> {jax...})" or
    # "... -> tensor<2x768xf32> {". Grab the first tensor after the top-level "->".
    m = _re.search(r"->\s*\(?\s*tensor<([0-9x]+)x(?:f32|i32|f16|bf16)>", text)
    if not m:
        return None
    return [int(d) for d in m.group(1).split("x")]


def test_gemma_from_obj(
    compile_only,
    base_dir_for_artifacts,
    inputs_dir,
    linalg_file,
    weights_file,
    metadata_file,
    tokenizer_dir,
    gemma_options_file,
):
    from transformers import AutoTokenizer
    from triton.backends.qcom_hexagon_backend.stablehlo_hexagon_launcher import (
        StableHLOHexagonLauncher,
    )
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    # Resolve paths from the pytest flags (set by run_gemma.py). When only
    # --linalg is given, weights/metadata/tokenizer are discovered next to it.
    global LINALG_FILE, WEIGHTS_FILE, METADATA_FILE, TOKENIZER_DIR, SEQ_LEN, BATCH_SIZE
    if linalg_file is not None:
        LINALG_FILE = Path(linalg_file)
        _infer_seq_shape(LINALG_FILE)
        _linalg_dir = LINALG_FILE.parent
        # Discover weights/metadata/tokenizer alongside the linalg file when the
        # explicit flags below don't override them.
        _sf = next(iter(_linalg_dir.glob("*.safetensors")), None)
        _meta = next(iter(_linalg_dir.glob("*.weights_metadata.json")), None)
        if _sf is not None:
            WEIGHTS_FILE = _sf
        if _meta is not None:
            METADATA_FILE = _meta
        _tok_dir = _linalg_dir / "fast_model_dir"
        if _tok_dir.is_dir():
            TOKENIZER_DIR = _tok_dir
    if weights_file is not None:
        WEIGHTS_FILE = Path(weights_file)
    if metadata_file is not None:
        METADATA_FILE = Path(metadata_file)
    if tokenizer_dir is not None:
        TOKENIZER_DIR = Path(tokenizer_dir)

    # The .o to run: GEMMA4_OBJ (set by the driver), required.
    _obj_env = os.environ.get("GEMMA4_OBJ")
    assert _obj_env, "GEMMA4_OBJ is not set — point it at a prebuilt kernel .o"
    obj_path = Path(_obj_env)

    gemma_options = (
        GemmaOptions.from_file(Path(gemma_options_file))
        if gemma_options_file is not None
        else GemmaOptions.from_object(obj_path)
    )
    options = HexagonOptions().__dict__
    options.update(gemma_options.runtime_options())
    options["enableProgressPrints"] = True
    # Synthetic inputs: fabricated on device by the wrapper. Unset GEMMA4_NO_INPUTS
    # to push real inputs instead (from --inputs-dir or the tokenizer path below).
    if os.environ.get("GEMMA4_NO_INPUTS", "0") != "0":
        options["syntheticInputs"] = True

    # The entry function of the model is @main (the .o exports _mlir_ciface_main).
    func_name = "main"

    # The linalg used only for the @main return signature; must match the one
    # the .o was lowered from. Prefer GEMMA4_LINALG, else the --linalg flag.
    _meta_linalg = os.environ.get("GEMMA4_LINALG") or (
        str(LINALG_FILE) if LINALG_FILE is not None else None
    )
    assert (
        _meta_linalg
    ), "No linalg for the return signature — set GEMMA4_LINALG or pass --linalg"
    meta_linalg = Path(_meta_linalg)

    assert obj_path.exists(), f"{obj_path} not found (GEMMA4_OBJ)"
    assert meta_linalg.exists(), (
        f"{meta_linalg} not found — needed only for the return signature; "
        f"set GEMMA4_LINALG to the linalg the .o was lowered from"
    )

    # Auto-detect the run mode from @main's return shape:
    #   rank-2 tensor<Bx768xf32>          => embedding model (cosine-similarity)
    #   rank-3 tensor<1x128x262144xf32>   => LLM logits (next-token decode)
    _ret_shape = _infer_return_shape(meta_linalg)
    embedding_mode = (
        _ret_shape is not None and len(_ret_shape) == 2 and _ret_shape[-1] == 768
    )
    if embedding_mode:
        print(
            f"==> MODE: EMBEDDING (return shape {_ret_shape} from {meta_linalg.name})",
            flush=True,
        )
    else:
        print(
            f"==> MODE: LLM LOGITS (return shape {_ret_shape} from {meta_linalg.name})",
            flush=True,
        )

    # Prompts for embedding mode: cosine similarity is computed pairwise across
    # them. Comma-separated via GEMMA4_PROMPTS; defaults to "dog", "cat".
    prompts = [
        p.strip()
        for p in os.environ.get("GEMMA4_PROMPTS", "dog,cat").split(",")
        if p.strip()
    ]

    # Load inputs before building the wrapper (needed for input profiles).
    # Two real-input sources: pre-generated .raw files (--inputs-dir) or the
    # tokenizer + weights path. Synthetic mode still builds real inputs here so
    # the wrapper can profile their shapes/dtypes.
    # --inputs-dir only counts if it actually holds a manifest.json.
    _has_manifest = bool(inputs_dir) and (Path(inputs_dir) / "manifest.json").exists()
    if _has_manifest:
        _idir = Path(inputs_dir)
        _manifest = json.loads((_idir / "manifest.json").read_text())
        inputs = []
        for i, entry in enumerate(_manifest):
            raw = (_idir / f"_mlir_ciface_main_t{i}.raw").read_bytes()
            dtype = getattr(torch, entry["dtype"])
            t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])
            inputs.append(t)
        print(f"Loaded {len(inputs)} inputs from {inputs_dir}", flush=True)
    elif embedding_mode:
        print(f"Loading weights and building embedding inputs for prompts: {prompts}")
        inputs = _build_embedding_inputs(prompts)
    else:
        print("Loading weights and building inputs...")
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        prompt = "What is the capital of France?"
        token_ids = tokenizer.encode(prompt)
        inputs = _build_inputs(token_ids)

    forward_meta, _ = _load_forward_metadata()
    _validate_program_inputs(inputs, forward_meta, meta_linalg)
    runtime_inputs_by_position = {
        location["position"]: inputs[index]
        for index, location in enumerate(forward_meta["input_locations"])
        if location["type_"] == "input_arg"
    }

    launcher = StableHLOHexagonLauncher()
    launcher_args = {
        "obj_path": obj_path,
        "mlir_meta_path": meta_linalg,
        "inputs": inputs,
        "func_name": func_name,
        "base_dir_for_artifacts": base_dir_for_artifacts,
        "iterations": 1,
        "options": options,
        "cleanup_device_post_exec": False,
    }

    if compile_only:
        launcher.run_from_obj(**launcher_args, compile_only=True)
        return

    # ------------------------------------------------------------------
    # Check the inputs before handing object linking and execution to the
    # launcher.
    # ------------------------------------------------------------------
    print("==> INPUT PRE-FLIGHT:", flush=True)
    print(
        f"      mode              : {'EMBEDDING (cosine similarity)' if embedding_mode else 'LLM logits'}",
        flush=True,
    )
    print(
        f"      syntheticInputs   : {options.get('syntheticInputs', False)} "
        f"(must be False for a meaningful similarity result)",
        flush=True,
    )
    if embedding_mode:
        print(f"      prompts           : {prompts}", flush=True)
        print(
            f"      expected out shape: [{BATCH_SIZE}, 768] (rows = per-prompt embeddings)",
            flush=True,
        )
    if 0 in runtime_inputs_by_position:
        input_ids = runtime_inputs_by_position[0]
        print(
            f"      input_ids   shape : {tuple(input_ids.shape)} dtype {input_ids.dtype}",
            flush=True,
        )
    if 1 in runtime_inputs_by_position:
        attention_mask = runtime_inputs_by_position[1]
        print(
            f"      attn_mask   shape : {tuple(attention_mask.shape)} "
            f"dtype {attention_mask.dtype}",
            flush=True,
        )
    print(
        f"      total inputs      : {len(inputs)} (@main ABI operands)",
        flush=True,
    )
    print("==> Starting StableHLO object launch...", flush=True)

    hex_outputs = launcher.run_from_obj(**launcher_args)
    if hex_outputs is None:
        raise RuntimeError("StableHLO launcher returned no outputs for device run")

    if embedding_mode:
        print("==> DECODE PATH: embedding / cosine-similarity", flush=True)
        import numpy as np

        embeddings = hex_outputs[0]
        if hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()
        embeddings = np.asarray(embeddings)
        print(f"Output Embedding Shape: {embeddings.shape}")

        valid = prompts[: embeddings.shape[0]]
        for i, p in enumerate(valid):
            print(f'\nPrompt [{i}]: "{p}"')
            print(f"Embedding Sample (first 5 values): {embeddings[i, :5]}")

        if len(valid) >= 2:
            print("\n--- Semantic Similarity Check ---")
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    e1, e2 = embeddings[i], embeddings[j]
                    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                    if n1 > 0 and n2 > 0:
                        sim = float(np.dot(e1, e2) / (n1 * n2))
                        print(
                            f'Cosine similarity between "{valid[i]}" '
                            f'and "{valid[j]}": {sim:.4f}'
                        )
        print("PASSED")
        return

    print("==> DECODE PATH: LLM logits / next-token", flush=True)
    logits = hex_outputs[0]
    print(f"Output shape: {logits.shape}")
    last_logits = logits[0, -1, :]
    next_token_id = int(last_logits.argmax())
    if not _has_manifest:
        next_token = tokenizer.decode([next_token_id])
        print(f"Next predicted token: {next_token!r}  (id={next_token_id})")
    else:
        print(f"Next predicted token id: {next_token_id}")
    print(f"Logits shape: {logits.shape}, max: {last_logits.max():.4f}")
    assert logits.shape == (
        1,
        SEQ_LEN,
        262144,
    ), f"Unexpected logits shape: {logits.shape}"
    print("PASSED")
