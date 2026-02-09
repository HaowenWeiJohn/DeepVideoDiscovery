# Serving Qwen3-Embedding Models Locally with vLLM

This guide covers how to host a Qwen3-Embedding model locally using vLLM and integrate it with the DVD pipeline. This gives you a self-hosted, OpenAI-compatible `/v1/embeddings` endpoint — the same interface used by `text-embedding-3-large` — so the existing codebase requires minimal changes.

## Prerequisites

- `vllm >= 0.8.5`
- GPU with sufficient VRAM (see model table below)

```bash
pip install "vllm>=0.8.5"
```

## Model Overview

| Model | HuggingFace ID | Max Embedding Dim | Max Context | Approx VRAM (FP16) |
|-------|---------------|-------------------|-------------|---------------------|
| **8B** | `Qwen/Qwen3-Embedding-8B` | 4096 | 32K | ~16 GB |
| **4B** | `Qwen/Qwen3-Embedding-4B` | 2560 | 32K | ~8 GB |
| **0.6B** | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32K | ~2 GB |

All three models share:
- 100+ language support
- Instruction-aware embeddings (optional, improves retrieval by 1-5%)
- Matryoshka Representation Learning (MRL) — you can truncate to smaller dimensions with minor quality loss

## Launching the Server

### Qwen3-Embedding-8B

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16
```

> **Known issue:** There is a [vLLM bug](https://github.com/vllm-project/vllm/issues/29496) where inputs at exactly `--max-model-len` cause hangs. Use 8192 (the HuggingFace-recommended length) instead of the model's maximum 32768.

### Qwen3-Embedding-4B

```bash
vllm serve Qwen/Qwen3-Embedding-4B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16
```

### Qwen3-Embedding-0.6B

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key Wf9i1OQCgKA4nAVZ__8CYJhfhExt_Yob120jbjIz0yA \
    --max-model-len 8192 \
    --dtype float16
```

### Selecting Specific GPUs

Use `CUDA_VISIBLE_DEVICES` to pin vLLM to specific GPU(s):

```bash
# Use only GPU 7
CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key Wf9i1OQCgKA4nAVZ__8CYJhfhExt_Yob120jbjIz0yA \
    --max-model-len 8192 \
    --dtype float16

# Use GPUs 6 and 7
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16 \
    --tensor-parallel-size 2
```

### Multi-GPU

Add `--tensor-parallel-size N` for multi-GPU setups (combine with `CUDA_VISIBLE_DEVICES` to pick which GPUs):

```bash
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16 \
    --tensor-parallel-size 2
```

### Shared GPU: `--gpu-memory-utilization`

If the GPU is already partially occupied (e.g., by the chat/VLM model), vLLM will fail with:

```
ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization (0.9, ...)
```

vLLM defaults to claiming 90% of **total** GPU memory. Use `--gpu-memory-utilization` to lower this:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16 \
    --gpu-memory-utilization 0.4
```

The value is a fraction of **total** GPU memory. For example, on a 95 GiB GPU with ~43 GiB free, `0.4` requests ~38 GiB which fits. For the 0.6B model you could go as low as `0.1`. Adjust based on your free memory.

### Key Flags Explained

| Flag | Purpose |
|------|---------|
| `--runner pooling` | **Required.** Tells vLLM to serve an embedding model (exposes `/v1/embeddings`). The older `--task embed` flag is deprecated since vLLM 0.11.0. |
| `--api-key` | Protects `/v1/*` endpoints with Bearer token auth. Can also be set via `VLLM_API_KEY` env var. |
| `--max-model-len` | Max input token length. Use 8192, not 32768 (see bug note above). |
| `--dtype float16` | Half precision. Use `bfloat16` if your GPU supports it natively. |
| `--gpu-memory-utilization` | Fraction of total GPU memory vLLM may claim (default `0.9`). Lower this when sharing a GPU with other processes. |

## API Usage

The server exposes an OpenAI-compatible endpoint at `POST /v1/embeddings`.

### curl

```bash
curl http://localhost:8888/v1/embeddings \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-secret-key" \
    -d '{
        "input": ["The capital of China is Beijing.", "Gravity is a fundamental force."],
        "model": "Qwen/Qwen3-Embedding-8B"
    }'
```

Response:
```json
{
    "data": [
        {"embedding": [0.0023, -0.0093, ...], "index": 0},
        {"embedding": [-0.0042, 0.0123, ...], "index": 1}
    ],
    "model": "Qwen/Qwen3-Embedding-8B",
    "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:8888/v1",
)

response = client.embeddings.create(
    input=["The capital of China is Beijing.", "Gravity is a fundamental force."],
    model="Qwen/Qwen3-Embedding-8B",
)

for item in response.data:
    print(f"Index {item.index}: dim={len(item.embedding)}")
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8888/v1/embeddings",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-secret-key",
    },
    json={
        "input": ["The capital of China is Beijing."],
        "model": "Qwen/Qwen3-Embedding-8B",
    },
)

embeddings = response.json()["data"]
```

## Instruction-Aware Embeddings (Optional)

All Qwen3-Embedding models are instruction-aware. For retrieval tasks, prepending a task instruction to **queries** (not documents) improves quality:

```python
def format_query(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"

# Queries: add instruction prefix
queries = [
    format_query("Given a video caption, retrieve relevant clips", "car chase scene"),
]

# Documents: NO prefix
documents = [
    "[From 00:12:30 to 00:12:40] A high-speed car chase through city streets...",
]
```

Write the instruction in **English** even for multilingual content — this gives the best results across all languages.

## DVD Integration

### Architecture

The embedding server runs as a **separate vLLM process** from your chat/VLM server. You will have two vLLM instances:

| Server | Model | Endpoint | Purpose |
|--------|-------|----------|---------|
| vLLM Chat | `Qwen3-VL-235B-A22B-Instruct-FP8` | `https://api.dd.works/v1` | Captioning, reasoning, frame inspect |
| vLLM Embedding | `Qwen/Qwen3-Embedding-8B` | `http://localhost:8888/v1` | All embedding calls (database build + retrieval) |

A single vLLM instance with `--runner pooling` **cannot** simultaneously serve chat completions. These must be separate processes.

### Config Changes (`dvd/config.py`)

Add embedding configuration to the VLLM block:

```python
elif SERVER == "VLLM":
    # ... existing chat model config ...

    # Embedding model (self-hosted)
    EMBEDDING_ENDPOINT = "http://localhost:8888/v1"
    EMBEDDING_API_KEY = "your-secret-key"
    AOAI_EMBEDDING_LARGE_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
    AOAI_EMBEDDING_LARGE_DIM = 4096
```

Adjust `AOAI_EMBEDDING_LARGE_DIM` based on your model choice:

| Model | `AOAI_EMBEDDING_LARGE_DIM` |
|-------|---------------------------|
| 8B | 4096 |
| 4B | 2560 |
| 0.6B | 1024 |

### Code Changes (`dvd/utils.py`)

In `AzureOpenAIEmbeddingService.get_embeddings()`, replace the hardcoded OpenAI endpoint:

```python
if api_key:
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + api_key
    }
    endpoint = getattr(config, 'EMBEDDING_ENDPOINT', 'https://api.openai.com/v1')
    url = f"{endpoint}/embeddings"
```

And update the auth header to use the embedding-specific key:

```python
api_key = getattr(config, 'EMBEDDING_API_KEY', config.OPENAI_KEY)
```

### Re-embedding Requirement

Switching embedding models changes the vector dimension (3072 -> 4096/2560/1024). **Existing databases must be re-built** since you cannot mix dimensions in the same `NanoVectorDB` instance. Delete or move old `database.json` files before re-running the pipeline.

## Running in the Background

By default, vLLM runs in the foreground and dies when you close the terminal. Here are three ways to keep it running persistently.

### Option 1: `nohup` + background (simplest)

```bash
CUDA_VISIBLE_DEVICES=7 nohup vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8888 \
    --api-key your-secret-key \
    --max-model-len 8192 \
    --dtype float16 \
    > vllm_embedding.log 2>&1 &
```

- Survives terminal close. Logs go to `vllm_embedding.log`.
- Check logs: `tail -f vllm_embedding.log`
- Stop: `kill $(pgrep -f "Qwen3-Embedding")`

### Option 2: `tmux` / `screen` (interactive)

```bash
tmux new -s embedding
# run the vllm serve command inside, then detach with Ctrl+B, D
```

- Reattach later: `tmux attach -t embedding`
- Stop: reattach and press `Ctrl+C`
- You keep full interactive access to logs.

### Option 3: `systemd` service (production)

Create a service file for auto-start on boot and auto-restart on crash:

```bash
sudo tee /etc/systemd/system/vllm-embedding.service << 'EOF'
[Unit]
Description=vLLM Embedding Server (Qwen3-Embedding-0.6B)
After=network.target

[Service]
Type=simple
Environment="CUDA_VISIBLE_DEVICES=7"
ExecStart=/home/hwjwei/miniconda3/envs/dvd/bin/vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --host 0.0.0.0 --port 8888 --api-key your-secret-key --max-model-len 8192 --dtype float16
Restart=on-failure
User=hwjwei

[Install]
WantedBy=multi-user.target
EOF
```

```bash
sudo systemctl daemon-reload
sudo systemctl start vllm-embedding    # start
sudo systemctl stop vllm-embedding     # stop
sudo systemctl status vllm-embedding   # check status
journalctl -u vllm-embedding -f        # tail logs
sudo systemctl enable vllm-embedding   # auto-start on boot
```

### Comparison

| Method | Survives terminal close | Auto-restart on crash | Auto-start on boot | Ease |
|--------|------------------------|----------------------|-------------------|------|
| `nohup &` | Yes | No | No | Easiest |
| `tmux`/`screen` | Yes | No | No | Easy, interactive |
| `systemd` | Yes | Yes | Yes | Requires sudo |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server hangs on certain inputs | Reduce `--max-model-len` (e.g., 8192 -> 4096). See [vLLM#29496](https://github.com/vllm-project/vllm/issues/29496). |
| `--task embed` not recognized | Upgrade to vLLM >= 0.11.0 and use `--runner pooling` instead. |
| Model loaded as generative | You forgot `--runner pooling`. |
| OOM on GPU | Add `--gpu-memory-utilization 0.4` (or lower) when sharing a GPU. Also try `--dtype float16`, reduce `--max-model-len`, or switch to a smaller model (4B/0.6B). |
| Auth errors | Ensure the `Authorization: Bearer <key>` header matches the `--api-key` value on the server. |
