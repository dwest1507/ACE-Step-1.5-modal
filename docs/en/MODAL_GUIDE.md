# Deploying ACE-Step 1.5 on Modal

[Modal](https://modal.com/) provides a serverless platform to deploy machine learning models. This is useful if you do not have a GPU on your local machine, or if you want to deploy the model for others to use. By following this guide, you can quickly spin up an ACE-Step 1.5 REST API endpoint on Modal using an A100, A10G or L4 GPU. This allows for scale-to-zero serverless execution. You can set up a free account with an initial free compute budget per month. You can also pay for additional compute, if desired.

*Costs incurred are solely your responsibility. Be sure you understand the [costs](https://modal.com/pricing) associated with using Modal before deploying.*

## Prerequisites

1. Create a [Modal account](https://modal.com/).
2. Authenticate with your Modal account:
   ```bash
   uv run modal setup
   ```

## Configuration

By default, the deployment uses the standard `turbo` DiT model (`acestep-v15-turbo`) with the `1.7B` LM model (`acestep-5Hz-lm-1.7B`), which provides a good balance between quality, speed, and VRAM usage. This fits comfortably on an A10G (24 GB).

### Choosing a DiT Model

The DiT model controls audio generation quality. Set `ACESTEP_CONFIG_PATH` in your `.env`:

| DiT Model | VRAM (weights) | Quality | Speed | `.env` value |
|-----------|---------------|---------|-------|-------------|
| `acestep-v15-turbo` (default) | ~4.7 GB | Very High | Fast (8 steps) | `ACESTEP_CONFIG_PATH=acestep-v15-turbo` |
| `acestep-v15-sft` | ~4.7 GB | High | Slower (50 steps) | `ACESTEP_CONFIG_PATH=acestep-v15-sft` |
| `acestep-v15-xl-turbo` | ~9 GB | Very High | Fast (8 steps) | `ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo` |
| `acestep-v15-xl-sft` | ~9 GB | Very High | Slower (50 steps) | `ACESTEP_CONFIG_PATH=acestep-v15-xl-sft` |
| `acestep-v15-xl-base` | ~9 GB | High | Slower (50 steps) | `ACESTEP_CONFIG_PATH=acestep-v15-xl-base` |

The **XL (4B DiT)** models offer higher audio quality but require more VRAM (~9 GB for weights alone vs ~4.7 GB for standard models). They need at least 20 GB recommended, making A10G (24 GB) the minimum GPU for XL.

> [!NOTE]
> The image build pre-downloads only `acestep-v15-turbo` (it ships in the main
> `ACE-Step/Ace-Step1.5` bundle) and, when the name contains `xl`, the matching XL repo.
> Choosing `acestep-v15-sft` therefore leaves its weights out of the image, and the
> container has to fetch them on the first cold start. Prefer the default or an XL model
> unless you are prepared for that.

### Choosing an LM Model

The LM model handles lyric/caption generation. Set `ACESTEP_LM_MODEL_PATH` in your `.env`:

- `acestep-5Hz-lm-0.6B` -- Lightweight, good for low-VRAM setups
- `acestep-5Hz-lm-1.7B` (default) -- Best balance of quality and speed
- `acestep-5Hz-lm-4B` -- Highest quality, requires A100

### Uploading Configuration

Define your choices in `.env`, then upload as a Modal Workspace Secret:

```bash
uv run modal secret create ace-step-api-secrets --from-dotenv .env --force
```
*(This creates or updates a secure secret grouping in your Modal dashboard called "ace-step-api-secrets" containing all the key-value pairs from your file.)*

*(Note: If you are using a gated model on Hugging Face, you'll need to create a Modal Secret named `huggingface-secret` containing your `HF_TOKEN`, or include `HF_TOKEN` in your `.env` before running the command above.)*

> [!IMPORTANT]
> Two variables in the `.env` you upload need attention, because a Modal Secret
> overrides the image's own defaults:
>
> - **Set `ACESTEP_LM_BACKEND=pt`.** `.env.example` ships `vllm`, and the deployment
>   requires the PyTorch backend: nanovllm's internals are CRIU-incompatible, so a
>   container that loads it cannot be restored from a GPU memory snapshot.
> - **Leave `ACESTEP_CHECKPOINTS_DIR` unset.** It is meant for sharing one model
>   directory across several local installs, and setting it redirects checkpoint
>   lookup away from `/workspace/checkpoints` — where the image build already placed
>   the weights — so the container would re-download several GB on every cold start.
>   `modal_app.py` pins `ACESTEP_PROJECT_ROOT=/workspace` for the same reason; do not
>   override it.

`modal_app.py` also bakes the deploy-time values of `ACESTEP_CONFIG_PATH`,
`ACESTEP_LM_MODEL_PATH` and `ACESTEP_LM_BACKEND=pt` into the image as defaults, so a
Secret that omits them still loads the models the image was built with. Values you do
set in the Secret take precedence at runtime — which is why the `ACESTEP_LM_BACKEND`
note above matters.

If the LM fails to load during the snapshot build, the server reports it as
unavailable rather than retrying inside a request: a request-time reload would pick
its own backend (`GenerateMusicRequest.lm_backend` defaults to `vllm` regardless of
`ACESTEP_LM_BACKEND`) and pull several GB while a caller waits. Generations that need
the LM then fail with a message pointing back here; redeploy so the snapshot captures
it.

### GPU Auto-Selection

The GPU is automatically selected based on your model combination:

| Configuration | GPU | VRAM |
|--------------|-----|------|
| Standard DiT + 0.6B LM | L4 | 24 GB |
| Standard DiT + 1.7B LM | A10G | 24 GB |
| XL DiT + any LM (except 4B) | A10G | 24 GB |
| Any DiT + 4B LM | A100 | 40/80 GB |

*Caution: Compute costs increase with GPU tier. Please refer to the [Modal pricing page](https://modal.com/pricing) for more information.*

## Deploying

To deploy the API, simply run the deployment command from the root of the repository:

```bash
uv run modal deploy modal_app.py
```

This will build the Docker container image, download the initial model weights to the container to ensure fast cold starts, and expose the application via a generated web URL.

## Using the API

You can make requests to your newly deployed API endpoint via `curl` or any other HTTP client.

To submit a full text-to-music configuration using one of the included example JSON payloads (like `examples/text2music/example_01.json`), use the `release_task` endpoint:

```bash
curl -X POST "<YOUR-DEPLOYED-URL>/release_task" \
     -H "Content-Type: application/json" \
     -d @examples/text2music/example_01.json
```

Two helper endpoints exist to build a payload for `/release_task`. Neither generates
audio on its own.

`/create_random_sample` returns one of the bundled example payloads at random, for
pre-filling a UI form. It takes `sample_type` (`simple_mode` or `custom_mode`):

```bash
curl -X POST "<YOUR-DEPLOYED-URL>/create_random_sample" \
     -H "Content-Type: application/json" \
     -d '{"sample_type": "simple_mode"}'
```

`/v1/create_sample` is the one that uses the LM: give it a free-form `query` and it
returns a generated caption, lyrics, BPM, key and time signature you can then post to
`/release_task`. It accepts `query`, `instrumental`, `vocal_language` and `temperature`:

```bash
curl -X POST "<YOUR-DEPLOYED-URL>/v1/create_sample" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "Create an upbeat rock track with heavy guitars",
         "vocal_language": "en"
     }'
```

`/release_task` is asynchronous. It returns `{"task_id": ..., "status": "queued",
"queue_position": ...}` — not the audio. Poll `/query_result` with that id until the
job finishes:

```bash
curl -X POST "<YOUR-DEPLOYED-URL>/query_result" \
     -H "Content-Type: application/json" \
     -d '{"task_id_list": ["<TASK-ID-FROM-RELEASE-TASK>"]}'
```

Each entry carries a `status` — `0` queued or running, `1` succeeded, `2` failed — and a
`result` field holding a **JSON-encoded string**, which you must parse before reading it.
Once `status` is `1`, the decoded result contains:

- `first_audio_path` — a *relative URL*, already in the form
  `/v1/audio?path=<url-encoded path>`. Append it to your deployment's base URL to
  download; do not pass it to `/v1/audio` a second time.
- `audio_paths` — the same relative URLs for every track in the batch.
- `raw_audio_paths` — the underlying container paths (for example
  `/workspace/.cache/acestep/tmp/api_audio/2c9c279c-....mp3`), if you would rather build
  the query string yourself. The UUID is assigned per generation.

If you do not receive a response, check the App Logs in the Modal dashboard.

Please note, the first time you deploy the model, it will take a few minutes to download the model weights to the container. Subsequent deployments will be much faster as the model weights will be cached. Also, your modal deployment will scale to zero when not in use, so you will not be charged for compute time when the model is not in use. But as a result, there will be cold starts for your first request after a 5-minute period of inactivity. This will result in a longer response time for the first request (around 1 minute). Subsequent requests will be faster as the model will be cached. This can all be customized in modal_app.py if you desire. This default behavior is optimized for cost savings not performance.

To download the track, append `first_audio_path` to your base URL as-is:

```bash
# first_audio_path == "/v1/audio?path=%2Fworkspace%2F.cache%2Facestep%2Ftmp%2F..."
curl -o downloaded_track.mp3 "<YOUR-DEPLOYED-URL><FIRST-AUDIO-PATH>"
```

Or build the query yourself from `raw_audio_paths`, URL-encoding the path:

```bash
curl -o downloaded_track.mp3 --get "<YOUR-DEPLOYED-URL>/v1/audio" \
     --data-urlencode "path=/workspace/.cache/acestep/tmp/api_audio/2c9c279c-....mp3"
```

> [!NOTE]
> `<YOUR-DEPLOYED-URL>` throughout this section is the `*.modal.run` URL that
> `uv run modal deploy modal_app.py` prints when the deploy finishes; it is also shown
> on the app's page in the Modal dashboard. Copy it from there rather than assembling it
> by hand.

For a full list of API endpoints, refer to the [REST API Guide](./API.md).
