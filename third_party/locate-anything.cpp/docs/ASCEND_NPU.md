# Ascend NPU (CANN) support

locate-anything.cpp runs on Huawei Ascend NPUs through ggml's
[CANN backend](https://github.com/ggml-org/ggml/tree/master/src/ggml-cann).
Validated on **Ascend 910B4** (Kylin Linux Advanced Server V10, aarch64,
CANN 8.2.RC1 host driver) — same detections as the CPU path, run on the NPU.

## How it plugs in

There is **no Ascend-specific code** in locate-anything. The backend selector in
`src/backend.cpp` walks the ggml device registry and picks the first
GPU-class device; the CANN backend reports its devices as
`GGML_BACKEND_DEVICE_TYPE_GPU`, so it is selected automatically once compiled in.
Per-op CPU fallback (via `ggml_backend_sched`) already handles any op the NPU
lacks a kernel for. The only build wiring added is one CMake switch:

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `LA_GGML_CANN` | OFF | Forward `GGML_CANN` to the ggml submodule (Ascend NPU) |

## Build

The build needs the CANN toolkit (compiler + runtime). Two flags differ from a
stock build:

- `-DSOC_TYPE=Ascend910B4` — ggml-cann otherwise auto-detects the SOC by running
  `npu-smi`, which isn't available at build time inside a container. Set it to
  your chip (`Ascend910B4`, `Ascend310P3`, ...).
- `LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64` — ggml-cann marks the CANN `lib64`
  link-search dir PRIVATE, so the final `locate-anything-cli` link can't find
  `-lascendcl` without this on the environment.

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/lib64:$LIBRARY_PATH"
cmake -B build -DLA_GGML_CANN=ON -DSOC_TYPE=Ascend910B4 -DLA_BUILD_CLI=ON
cmake --build build -j
```

The container build (`Dockerfile.cann`) bakes both in.

## Model precision: use f16/f32, not the q-quants

The CANN weight-quantized matmul (`aclnnWeightQuantBatchMatmulV2`) requires a
contiguous output tensor and aborts on this model's graph:

```
CANN error: AclNN_Parameter_Error(EZ1001): only support y tensor is contiguous.
  in function ggml_cann_mul_mat_quant at ggml-cann/aclnn_ops.cpp
```

So `q8_0`/`q6_k`/`q5_k`/`q4_k` GGUFs run **CPU-only**. For the NPU, use the
**f16** GGUF (≈8.6 GB; the precision-matched mode) or f32 (≈15 GB). Build it from
the f32 GGUF:

```sh
locate-anything-cli quantize models/locate-anything-f32.gguf \
    models/locate-anything-f16.gguf f16
```

## Run (Docker)

`Dockerfile.cann` + `docker-compose.npu.yml` package it. The container maps one
NPU plus the Ascend manager devices and mounts the host driver read-only:

```sh
docker build -f Dockerfile.cann -t locate-anything-npu:latest .

docker run --rm --privileged \
  --device /dev/davinci1 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=1 -e LA_DEVICE=CANN0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$PWD/models:/models" -v "$PWD/data:/data" \
  locate-anything-npu:latest detect \
    --model /models/locate-anything-f16.gguf \
    --input /data/parity_image.png \
    --prompt "Locate all the instances that matches the following description: cat</c>remote."
```

`LA_DEVICE` selects the device: unset/`CANN0` → first NPU, `cpu` → force CPU.

## Verification (910B4)

`parity_image.png`, prompt `cat</c>remote.`, `--mode hybrid`, f16 GGUF:

| device | detections | boxes |
| ------ | ---------- | ----- |
| CPU (q8_0) | 2× cat, 2× remote | reference |
| NPU CANN0 (f16) | 2× cat, 2× remote | match within ~1–2 px (f16 vs q8 drift) |

`npu-smi info` during inference shows the `locate-anything` process resident on
the NPU (≈7.9 GB HBM) with AICore active (~44% sampled) — the matmuls execute on
the Ascend cores, not the CPU.
