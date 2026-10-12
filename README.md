# llm-deploy

Standalone, profile-driven LLM deployment using **Apptainer for both local
execution and single-job Slurm execution**. No GenAI-Net imports or cluster
settings are required for the local trial.

**Status:** implementation and tests are prepared but have not been runtime-tested.
No trial, image pull, model download, or deployment was run during preparation.
Existing GenAI-Net deployment scripts and services are unaffected.

## Ubuntu + TITAN RTX trial

Requirements on the trial machine:

- Linux x86-64, Python 3.10+ with venv support, and Apptainer on PATH.
- A working NVIDIA driver compatible with the container's CUDA runtime;
  `nvidia-smi` should recognize the TITAN RTX.
- Internet access to GHCR and Hugging Face for initial setup, and enough disk
  space for the container, conversion temporary files, and model cache.

No host CUDA toolkit, llama.cpp build, Docker, Slurm, or host PyTorch/vLLM
installation is needed. The engine and its GPU software dependencies are in
the container; the host still supplies the NVIDIA driver. GPU compatibility
will be checked by the opt-in trial, not assumed from the model size.

Copy/extract this directory or `llm-deploy.tar.gz` on the other machine, then:

```bash
cd llm-deploy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
apptainer --version
nvidia-smi

# Inspect without downloading, writing caches, or starting a process:
llm-deploy plan --profile tiny-qwen

# Only when ready to run the real GPU trial:
python scripts/trial.py --run --device 0
```

Without `--run`, the trial only describes what it would do. Nothing runs
on installation or test discovery. Use `--apptainer /path/to/apptainer` if
Apptainer is not on PATH.

The `tiny-qwen` profile uses **Qwen2.5-0.5B-Instruct Q4_K_M**: a single
491,400,032-byte GGUF (about 491 MB / 469 MiB), pinned by revision and SHA-256.
It requests a 2,048-token context, one sequence, small batches, F16 KV cache,
no flash attention, and **no speculative decoding**. This is an infrastructure
smoke-test model, not a replacement for the larger model's reasoning quality.

The trial pulls/converts the pinned container if needed, downloads/verifies the
GGUF, starts a loopback service on port 18089, checks model identity and chat
completion, and requires evidence of GPU layer offload in the engine log. It
records GPU/driver information, command, response, and log paths under
`.trial/<timestamp>/`, then shuts down its own server. It refuses an occupied
port. The 30-minute default timeout includes initial image conversion and
model download; raise `--timeout` if initial setup needs longer.

## Keep a local server running

```bash
llm-deploy start --executor local --profile tiny-qwen --device 0
```

This performs the same container/model preparation, then serves in the foreground.
It prints `Ready` after health, model identity, and chat checks pass. Keep the
terminal open; Ctrl-C stops the server and its workers.

In another terminal with the virtualenv active:

```bash
llm-deploy health
llm-deploy smoke --profile tiny-qwen
curl http://127.0.0.1:18081/v1/models
```

Client settings:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18081/v1
export OPENAI_API_KEY=not-used
export OPENAI_MODEL=tiny-qwen
```

Pass the model variable explicitly as the request's `model` field. The service
binds to loopback and does not enable authentication. To use another client
machine, run there (replace username/hostname):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18081:127.0.0.1:18081 USER@INFERENCE_HOST
```

## Apptainer and artifact storage

Both executors use the same `apptainer exec --nv --cleanenv` path. A local
`--device` selection, or Slurm's assigned `CUDA_VISIBLE_DEVICES`, is forwarded
into the container explicitly. `--cpu` disables llama.cpp GPU offload and omits
`--nv`; successful CPU inference is not evidence of GPU compatibility.

By default, artifacts live in `~/.cache/llm-deploy` (respecting `XDG_CACHE_HOME`).
Choose another filesystem with `--cache-dir /path/to/storage`.

- `containers/`: SIF images keyed by immutable OCI digest, with a lock and local
  SHA-256 record. Subsequent launches verify and reuse the converted image.
- `models/`: pinned HF snapshots/GGUF files, mounted read-only in the container.
- `compile/`: writable caches isolated by the full resolved profile.
- `runs/`: per-run profile, launch command, engine log, and startup response.
- `apptainer-cache/`: OCI layers used during pulls.

Apptainer's normal temporary directory is used for conversion. Set
`APPTAINER_TMPDIR` to a suitable disk if needed; inside Slurm, `SLURM_TMPDIR` is
used when available and no override was given. The host Python launcher handles
HF downloads and GGUF checksums; inference runs inside Apptainer. The pinned
OCI digest identifies the upstream image, while the recorded SIF checksum
verifies reuse of its locally converted representation.

## Profiles and ordinary decoding defaults

```bash
llm-deploy profiles
llm-deploy validate --profile tiny-qwen
llm-deploy start --profile ./my-model.yaml
```

Profiles are data-only YAML validated using the bundled JSON Schema. Unknown
fields, floating model revisions, and options for the wrong engine are rejected.
Model profiles are independent of local/Slurm machine settings.

| Setting | llama.cpp default | vLLM default |
| --- | --- | --- |
| Speculative decoding | Off | Off |
| Context tokens | 2,048 | 4,096 |
| Concurrent sequences | 1 | 1 |
| Precision | Determined by GGUF | `auto` |
| GPU settings | Up to 99 layers | One tensor-parallel rank, 85% memory utilization |
| Flash attention | Off | Engine default |
| Reasoning/tool parsers | Not injected | Not injected |

These are conservative starting points, not universal hardware guarantees.
For llama.cpp the context allocation is shared among slots; one slot receives
the whole configured context. Omit `speculative` entirely for target-only
inference: no draft is downloaded or mounted as a separate model argument.
Speculation currently supports vLLM DFlash with a matched draft checkpoint.

For another model, copy `examples/custom-gguf.yaml` or `examples/custom-vllm.yaml`.
Change the name, repository, pinned revision, and (for GGUF) filename, bytes, and
SHA-256. The engine must support the architecture and chat template. This initial
version supports one-file GGUFs and safetensors snapshots; HF shard validation
uses the checkpoint's index rather than a fixed shard count. HF authentication
uses the host client's login or `HF_TOKEN`; do not place tokens in profiles.

The engine selects a default immutable container image:

- llama.cpp: the original CUDA 12 `server-cuda12-b10630` image.
- vLLM: the original `vllm/vllm-openai` v0.28.0 image.

`validate` and `plan` show the full resolved image digest. To use a different
engine release, specify a digest-pinned image in the profile:

```yaml
container:
  image: docker://REGISTRY/REPOSITORY@sha256:REPLACE_WITH_64_HEX_DIGEST
```

The image must provide `/app/llama-server` for llama.cpp or `vllm` on PATH for
vLLM, and accept the generated CLI flags. The example above is a placeholder
and fails validation until replaced. An image change may be needed for another
GPU, architecture, or runtime version.

`qwen-h200` and `qwen-dflash` retain the larger target model without/with DFlash2.
Their memory needs differ substantially from the small trial. They use the
original pinned vLLM container but have not been tested via this new launcher.

## Slurm: one submission, same Apptainer runtime

```bash
llm-deploy start --executor slurm --profile tiny-qwen \
  --cache-dir /shared/scratch/llm-deploy \
  --partition YOUR_PARTITION --account YOUR_ACCOUNT \
  --gres gpu:1 --cpus 4 --mem 8G --time 02:00:00
```

If your cluster uses modules, add `--apptainer-module apptainer/1.5.2` (or the
module provided by your site). It is loaded in the batch login shell. For local
execution, load any needed module in your own shell before launching.

The package virtualenv and cache must be accessible on the compute node; no
native inference engine installation is required. Container conversion, model
downloads, and serving occur **within one allocation and job ID**. There are no
nested submissions. The profile is embedded in the job so queued runs cannot
pick up later edits. GPU visibility comes from Slurm; do not pass `--device`.
The sample resources are for the tiny model, not the H200 profiles.

Setup counts toward job wall time. Submission prints the job ID; use `squeue`
and `scancel JOB_ID` for management. Logs use Slurm's default output location.
`llm-deploy plan --executor slurm` prints the generated script without submitting.
For loopback access on a compute node, use an SSH tunnel to that node (with
ProxyJump if permitted), or explicitly configure `--host` for your cluster's
approved access method. No cluster hostname is hard-coded.

## Tests and standalone repository

Tests are supplied **but were not run during preparation**:

```bash
# Offline unit tests, without containers/GPU/downloads/submission:
python -m unittest discover -s tests -v

# Opt-in GPU/container integration trial on the other machine:
python scripts/trial.py --run
```

Copy this whole directory to the destination and use `git init` and `git add .`
to create an independent repository. It includes package metadata, schema,
profiles, docs, tests, trial script, and the existing GPLv3 license. No parent
repository files are required; no remote has been created. Engine and model
artifacts retain their upstream licenses. Weights, containers, credentials,
virtualenvs, and logs do not belong in source control.

To build a wheel on the destination:

```bash
python -m pip install build
python -m build
```

The wheel includes profiles and schema; the source archive includes the trial.

Sources: [small GGUF model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF),
[pinned model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67).
