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

By default, the deployment uses the `1.7B` LM model (`acestep-5Hz-lm-1.7B`), which provides a good balance between speed and VRAM usage. This allows the model to comfortably fit within common Modal GPUs like the A10G.

If you wish to configure the Language Model or other environment variables, you should define them in your `.env` file (e.g., `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B`), and then upload this configuration as a Modal Workspace Secret.

Run the following command in the project root:
```bash
uv run modal secret create ace-step-api-secrets --from-dotenv .env --force
```
*(This creates or updates a secure secret grouping in your Modal dashboard called "ace-step-api-secrets" containing all the key-value pairs from your file.)*

*(Note: If you are using a gated model on Hugging Face, you'll need to create a Modal Secret named `huggingface-secret` containing your `HF_TOKEN`, or include `HF_TOKEN` in your `.env` before running the command above.)*

The GPU configuration in Modal will be automatically selected based on the LM model size. The table below shows the GPU configuration for each LM model:

| LM Model | GPU Configuration |
|----------|-------------------|
| `acestep-5Hz-lm-0.6B` | `L4` |
| `acestep-5Hz-lm-1.7B` | `A10G` |
| `acestep-5Hz-lm-4B` | `A100` |

*Caution: Compute costs will increase with the size of the model. Please refer to the [Modal pricing page](https://modal.com/pricing) for more information.*

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

To download this file from your Modal app to your local machine, use the `/v1/audio` endpoint and pass this exact path:

```bash
curl -o downloaded_track.mp3 "https://<YOUR-MODAL-WORKSPACE>--acestep-api-fastapi-app.modal.run/v1/audio?path=/workspace/.cache/acestep/tmp/api_audio/2c9c279c-b3a8-42f0-a3e9-cf2ee2dbd021.mp3"
```

*(Note: Replace the URL with the exact URL provided by `uv run modal deploy`.)*

For a full list of API endpoints, refer to the [REST API Guide](./API.md).
