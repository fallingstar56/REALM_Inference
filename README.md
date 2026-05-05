# Introduction
This repository is a fork of [REALM](https://github.com/martin-sedlacek/REALM.git). We add an interface for infering using the [GR00T](https://github.com/NVIDIA/Isaac-GR00T.git)(N1.6 & N1.7).

# Usage

## One-command evaluation scripts

The repository root contains two convenience launchers for running full REALM evaluation batches with GR00T:

- `run17.sh` runs REALM with GR00T N1.7.
- `run16.sh` runs REALM with GR00T N1.6.

Both scripts start the matching GR00T server on the host, wait until the server is reachable, start one REALM Docker container with host networking, and then run `examples/02_evaluate.py` for every configured task and perturbation pair. Server logs are written to `logs/gr00t_n17_server_*.log` or `logs/gr00t_n16_server_*.log`.

Before running either script, make sure Docker is available, the REALM dataset path exists, and the NVIDIA Omniverse EULA has been accepted. For non-interactive runs, set:

```bash
export OMNIVERSE_EULA_ACCEPTED=1
```

### Run GR00T N1.7 batch evaluation

`run17.sh` uses the GR00T N1.7 repository and model by default:

```bash
cd /home/xuhanyuan/REALM_Inference
./run17.sh
```

Default settings:

- GR00T repository: `/home/xuhanyuan/Isaac-GR00T`
- Model: `nvidia/GR00T-N1.7-3B`
- Embodiment tag: `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`
- Task IDs: `0 2 3 5 6 8 9`
- Perturbation IDs: `1 3 4 5 15`
- Repeats: `15`
- Max steps: `350`
- Horizon: `15`
- Evaluation model name/type: `gr00t_n17`

Useful options:

```bash
./run17.sh --no-uv-sync
./run17.sh --reuse-server
./run17.sh --help
```

Use `--no-uv-sync` when the GR00T N1.7 `uv` environment is already prepared. Use `--reuse-server` when a compatible N1.7 server is already listening on the configured host and port.

### Run GR00T N1.6 batch evaluation

`run16.sh` uses the GR00T N1.6 release repository and DROID model by default:

```bash
cd /home/xuhanyuan/REALM_Inference
./run16.sh
```

Default settings:

- GR00T repository: `/home/xuhanyuan/Isaac-GR00T-n1.6-release`
- Model: `nvidia/GR00T-N1.6-DROID`
- Embodiment tag: `OXE_DROID`
- Task IDs: `0 1 2 3 4 5 6 8 9`
- Perturbation IDs: `1 3 4 5 15`
- Repeats: `15`
- Max steps: `350`
- Horizon: `8`
- Evaluation model name: `gr00t_n16`
- Evaluation model type: `GR00T_N16`

Useful options:

```bash
./run16.sh --no-uv-sync
./run16.sh --reuse-server
./run16.sh --help
```

The N1.6 server is started with `--use_sim_policy_wrapper`, which is required for the flat DROID-style observation and action keys used by the REALM adapter. Unlike `run17.sh`, `run16.sh` does not automatically reuse an unknown process already listening on the target port, because an N1.7 server returns incompatible modality keys. Stop the existing process, choose another `GR00T_PORT`, or pass `--reuse-server` only when the running service is already a compatible N1.6 server.

### Common environment overrides

The launchers can be customized with environment variables:

```bash
export REALM_ROOT=/home/xuhanyuan/REALM_Inference
export REALM_DATA_PATH=/home/xuhanyuan/REALM_DATA
export REALM_DOCKER_IMAGE=realm
export GR00T_HOST=127.0.0.1
export GR00T_SERVER_HOST=0.0.0.0
export GR00T_PORT=5555
export REPEATS=15
export MAX_STEPS=350
export MODEL_SERVER_TIMEOUT=900
```

N1.7-specific overrides:

```bash
export GR00T_ROOT=/home/xuhanyuan/Isaac-GR00T
export GR00T_MODEL_PATH=nvidia/GR00T-N1.7-3B
export GR00T_EMBODIMENT_TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
export GR00T_DEVICE=cuda:0
```

N1.6-specific overrides:

```bash
export GR00T_N16_ROOT=/home/xuhanyuan/Isaac-GR00T-n1.6-release
export GR00T_N16_MODEL_PATH=nvidia/GR00T-N1.6-DROID
export GR00T_N16_EMBODIMENT_TAG=OXE_DROID
export GR00T_N16_DEVICE=cuda:0
```

## Run REALM with GR00T N1.7

### 1. Set environment variables on the host

Use the following paths before starting either GR00T or REALM:

```bash
export REALM_ROOT=/home/xuhanyuan/REALM_Inference
export GR00T_ROOT=/home/xuhanyuan/Isaac-GR00T
export REALM_DATA_PATH=/home/xuhanyuan/REALM_DATA
export HF_ENDPOINT=https://hf-mirror.com
export GR00T_MODEL_PATH=nvidia/GR00T-N1.7-3B
export GR00T_EMBODIMENT_TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
export GR00T_DEVICE=cuda:0
```

`GR00T_MODEL_PATH` can point either to a local checkpoint or to a Hugging Face model ID such as `nvidia/GR00T-N1.7-3B`.

### 2. Start the GR00T server

Move into the Isaac-GR00T repository and install the Python 3.10 environment:

```bash
cd /home/xuhanyuan/Isaac-GR00T
uv sync --python 3.10
```

On the first run, GR00T may fail with a `403 Client Error` when it tries to download `nvidia/Cosmos-Reason2-2B`. That model is gated.

Before retrying, do the following:

1. Open https://huggingface.co/nvidia/Cosmos-Reason2-2B and request access.
2. Make sure the mirror endpoint is enabled:

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. Log in from the Isaac-GR00T environment and paste a Hugging Face token that already has access:

   ```bash
   uv run huggingface-cli login
   ```

After access is approved, start the GR00T server:

```bash
export GR00T_MODEL_PATH=nvidia/GR00T-N1.7-3B
export GR00T_EMBODIMENT_TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
export GR00T_DEVICE=cuda:0

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$GR00T_MODEL_PATH" \
  --embodiment-tag "$GR00T_EMBODIMENT_TAG" \
  --host 0.0.0.0 \
  --port 5555 \
  --device "$GR00T_DEVICE"
```

Successful startup looks like this:

```text
✓ Server ready — listening on 0.0.0.0:5555
Server is ready and listening on tcp://0.0.0.0:5555
```

### 3. Start REALM in Docker

Make sure `GR00T_ROOT` points to the host checkout before starting the container:

- Start the Docker daemon first.
- The launcher will ask you to accept the NVIDIA Omniverse EULA on first start.

```bash
export GR00T_ROOT=/home/xuhanyuan/Isaac-GR00T
cd /home/xuhanyuan/REALM_Inference
./scripts/run_docker.sh --headless /home/xuhanyuan/REALM_DATA
```

The launcher uses `--network=host`, so the container can reach the GR00T server via `127.0.0.1:5555`.

### 4. Run evaluation inside the container

Inside the REALM container, run:

```bash
cd /app
export GR00T_ROOT=/app/Isaac-GR00T
export ISAAC_GR00T_ROOT=/app/Isaac-GR00T
export OMNIGIBSON_HEADLESS=1

pip install pyzmq==27.0.1

python examples/02_evaluate.py \
  --perturbation_id 0 \
  --task_id 1 \
  --repeats 10 \
  --max_steps 100 \
  --horizon 15 \
  --model_name gr00t_n17 \
  --model_type gr00t_n17 \
  --host 127.0.0.1 \
  --port 5555 \
  --experiment_name gr00t_n17_smoke \
  --save_mp4
```

### 5. Dataset check and common recovery step

If evaluation fails early, first check that `REALM_DATA_PATH/datasets/og_dataset` contains the full dataset tree:

- `objects/`
- `scenes/`
- `metadata/`
- `systems/`

If the target dataset directory is incomplete and only contains the Omniverse license file, restore it from the repository copy:

```bash
rm -rf /home/xuhanyuan/REALM_DATA/datasets/og_dataset
cp -a /home/xuhanyuan/REALM_Inference/data/datasets/og_dataset /home/xuhanyuan/REALM_DATA/datasets/
```

### 6. Common issues

- `403 Client Error` for `nvidia/Cosmos-Reason2-2B`: request access on Hugging Face first, then run `uv run huggingface-cli login` with a token that can access the gated model.
- Slow or blocked model downloads: keep `HF_ENDPOINT=https://hf-mirror.com` enabled before launching the GR00T server.
- GR00T not reachable from REALM: make sure the GR00T server is already listening on `0.0.0.0:5555` on the host before starting evaluation.
- No standalone videos saved: keep `--save_mp4` in the evaluation command.

## Run REALM with GR00T N1.6

GR00T N1.6 now uses the same single-server pattern as N1.7. REALM only adds the observation and action alignment layer; rollout and controller logic stay unchanged.

### 1. Set environment variables on the host

```bash
export REALM_ROOT=/home/xuhanyuan/REALM_Inference
export GR00T_N16_ROOT=/home/xuhanyuan/Isaac-GR00T-n1.6-release
export REALM_DATA_PATH=/home/xuhanyuan/REALM_DATA
export HF_ENDPOINT=https://hf-mirror.com
export GR00T_N16_MODEL_PATH=nvidia/GR00T-N1.6-DROID
export GR00T_N16_EMBODIMENT_TAG=OXE_DROID
export GR00T_N16_DEVICE=cuda:0
```

### 2. Start the GR00T N1.6 server

```bash
cd /home/xuhanyuan/Isaac-GR00T-n1.6-release
uv sync --python 3.10

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$GR00T_N16_MODEL_PATH" \
  --embodiment-tag "$GR00T_N16_EMBODIMENT_TAG" \
  --use_sim_policy_wrapper \
  --host 0.0.0.0 \
  --port 5555 \
  --device "$GR00T_N16_DEVICE"
```

`--use_sim_policy_wrapper` is required because REALM sends the flat DROID-style keys that the official N1.6 DROID example uses.

### 3. Start REALM in Docker

Before starting the container, export the N1.6 repo path so the client-side adapter can import the official `PolicyClient` inside Docker:

```bash
export GR00T_N16_ROOT=/home/xuhanyuan/Isaac-GR00T-n1.6-release
cd /home/xuhanyuan/REALM_Inference
./scripts/run_docker.sh --headless /home/xuhanyuan/REALM_DATA
```

### 4. Run evaluation inside the container

```bash
cd /app
export GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
export ISAAC_GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
export OMNIGIBSON_HEADLESS=1

pip install pyzmq==27.0.1

python examples/02_evaluate.py \
  --perturbation_id 0 \
  --task_id 0 \
  --repeats 1 \
  --max_steps 1000 \
  --horizon 8 \
  --model_name gr00t_n16 \
  --model_type GR00T_N16 \
  --host 127.0.0.1 \
  --port 5555 \
  --experiment_name gr00t_n16_smoke \
  --save_mp4
```

You can also launch through the unified script:

```bash
cd /home/xuhanyuan/REALM_Inference
./scripts/eval.sh -m gr00t_n16 -c "$GR00T_N16_MODEL_PATH" -e docker
```
