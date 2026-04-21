# REALM: A Real-to-Sim Validated Benchmark for Generalization in Robotic Manipulation

<p align="center">
  <a href="https://martin-sedlacek.com/realm"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.19562/"><img src="https://img.shields.io/badge/paper-preprint-red" alt="arXiv"></a>
  <a href="https://github.com/martin-sedlacek/REALM/wiki"><img src="https://img.shields.io/badge/doc-page-orange" alt="Documentation"></a>
  <a href="https://github.com/martin-sedlacek/REALM/issues"><img src="https://img.shields.io/github/issues/martin-sedlacek/REALM?color=yellow" alt="Issues"></a>
  <a href="https://github.com/martin-sedlacek/REALM/discussions"><img src="https://img.shields.io/github/discussions/martin-sedlacek/REALM?color=blueviolet" alt="Discussions"></a>
</p>

![](./images/realm_overview_fig.png)

REALM is a large-scale realistic simulation environment and benchmark for generalization 
in robotic manipulation. It supports 7 distinct manipulation skills and stress-tests them 
against 15 perturbations. Through empirical validation, we show that evaluation results 
in simulation are strongly correlated to real-world performance. 

# Introduction
This repository is a fork of [REALM](https://github.com/martin-sedlacek/REALM.git).We add an interface for infering using the [GR00T](https://github.com/NVIDIA/Isaac-GR00T.git)(N1.6 & N1.7).

# Usage
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
  --horizon 8 \
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