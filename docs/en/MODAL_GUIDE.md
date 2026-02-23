# Deploying ACE-Step 1.5 on Modal

Modal provides a serverless platform to deploy machine learning models. By following this guide, you can quickly spin up an ACE-Step 1.5 REST API endpoint on Modal using an A10G or L4 GPU. This allows for scale-to-zero serverless execution.

## Prerequisites

1. Create a [Modal account](https://modal.com/).
2. Install the Modal CLI on your local machine using `uv` (as recommended for this project):
   ```bash
   uv pip install modal
   ```
3. Authenticate with your Modal account:
   ```bash
   uv run modal setup
   ```

## Configuration

By default, the deployment uses the `1.7B` LM model (`acestep-5Hz-lm-1.7B`), which provides a good balance between speed and VRAM usage. This allows the model to comfortably fit within common Modal GPUs like the A10G.

If you wish to change the default Language Model, you can define an `ACESTEP_LM_MODEL` environment variable in Modal's Secrets.  
To change it via Secrets:
1. Go to the [Modal Secrets Dashboard](https://modal.com/secrets)
2. Create a generic Secret dictionary, add the key `ACESTEP_LM_MODEL` and the desired value (e.g., `acestep-5Hz-lm-0.6B` or `0.6B`).

*(Note: If you are using a gated model on Hugging Face, you'll need to create a Modal Secret named `huggingface-secret` with your `HF_TOKEN`.)*

## Deploying

To deploy the API, simply run the deployment command from the root of the repository:

```bash
uv run modal deploy modal_app.py
```

This will build the Docker container image, download the initial model weights to the container to ensure fast cold starts, and expose the application via a generated web URL.

## Using the API

You can make requests to your newly deployed API endpoint via `curl` or any other HTTP client.

To generate a task description and submit a synchronous text-to-music task:

```bash
curl -X POST "https://<YOUR-MODAL-WORKSPACE>--acestep-api-fastapi-app.modal.run/create_random_sample" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "Create an upbeat rock track with heavy guitars",
         "vocal_language": "en"
     }'
```

*(Note: Replace the URL with the exact URL provided by `uv run modal deploy`.)*

For a full list of API endpoints, refer to the [REST API Guide](./API.md).
