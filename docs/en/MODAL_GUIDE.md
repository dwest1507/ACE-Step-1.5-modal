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
> Leave `ACESTEP_CHECKPOINTS_DIR` unset in the `.env` you upload. It is meant for
> sharing one model directory across several local installs, and setting it
> redirects checkpoint lookup away from `/workspace/checkpoints` — where the image
> build already placed the weights — so the container would re-download several GB
> on every cold start. `modal_app.py` pins `ACESTEP_PROJECT_ROOT=/workspace` for the
> same reason; do not override it.

`modal_app.py` also bakes the deploy-time values of `ACESTEP_CONFIG_PATH`,
`ACESTEP_LM_MODEL_PATH` and `ACESTEP_LM_BACKEND=pt` into the image as defaults, so a
Secret that omits them still loads the models the image was built with. Values you do
set in the Secret take precedence at runtime.

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
curl -X POST "https://<YOUR-MODAL-WORKSPACE>--acestep-api-fastapi-app.modal.run/release_task" \
     -H "Content-Type: application/json" \
     -d @examples/text2music/example_01.json
```

Alternatively, to generate a task description and submit a synchronous text-to-music task from a simple prompt (leveraging the LM):

```bash
curl -X POST "https://<YOUR-MODAL-WORKSPACE>--acestep-api-fastapi-app.modal.run/create_random_sample" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "Create an upbeat rock track with heavy guitars",
         "vocal_language": "en"
     }'
```

In the JSON response returned by either of these API calls, you will find a `result` object containing a field named `first_audio_path`. This represents your generated track's location within the remote container (for example, `/workspace/.cache/acestep/tmp/api_audio/2c9c279c-b3a8-42f0-a3e9-cf2ee2dbd021.mp3`). The unique UUID is randomly assigned for every new audio generation task.

If you do not receive a response, check the App Logs in the Modal dashboard. The path should be listed there.

Please note, the first time you deploy the model, it will take a few minutes to download the model weights to the container. Subsequent deployments will be much faster as the model weights will be cached. Also, your modal deployment will scale to zero when not in use, so you will not be charged for compute time when the model is not in use. But as a result, there will be cold starts for your first request after a 5-minute period of inactivity. This will result in a longer response time for the first request (around 1 minute). Subsequent requests will be faster as the model will be cached. This can all be customized in modal_app.py if you desire. This default behavior is optimized for cost savings not performance.

To download this file from your Modal app to your local machine, use the `/v1/audio` endpoint and pass this exact path:

```bash
curl -o downloaded_track.mp3 "https://<YOUR-MODAL-WORKSPACE>--acestep-api-fastapi-app.modal.run/v1/audio?path=/workspace/.cache/acestep/tmp/api_audio/2c9c279c-b3a8-42f0-a3e9-cf2ee2dbd021.mp3"
```

*(Note: Replace the URL with the exact URL provided by `uv run modal deploy`.)*

For a full list of API endpoints, refer to the [REST API Guide](./API.md).
