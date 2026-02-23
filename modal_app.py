import os
import modal

# Create a Modal app
app = modal.App("acestep-api")

# Get defaults from environment variables
DEFAULT_LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "1.7B")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Definition of the models to download during image build
def download_models():
    import os
    from huggingface_hub import snapshot_download
    
    print("Downloading ACE-Step 1.5 models...")
    # The default repo
    snapshot_download(
        repo_id="ACE-Step/Ace-Step1.5", 
        local_dir="/workspace/checkpoints"
    )

    # Optional: download other models if ACESTEP_LM_MODEL is heavily customized
    # Since ACE-Step/Ace-Step1.5 already contains "acestep-v15-turbo" and "acestep-5Hz-lm-1.7B"
    # we usually don't need to specify others unless explicitly asked.

def configure_environment():
    # Pass down the LM model override to the app
    os.environ["ACESTEP_LM_MODEL"] = os.environ.get("ACESTEP_LM_MODEL", "1.7B")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("uv", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", remote_path="/workspace", ignore=[".git", ".venv", "**/.venv", "__pycache__", "**/*.pyc", "checkpoints", "logs"], copy=True)
    .workdir("/workspace")
    # Install dependencies according to the existing project
    .run_commands(
        "uv pip install --system torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "uv pip install --system -e ."
    )
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")] if HF_TOKEN else []
    )
)

@app.function(
    image=image,
    gpu="A10G", # Good balance of speed and VRAM usage
    scaledown_window=300, # Keep the container warm for 5 minutes
    timeout=3600,
    secrets=[modal.Secret.from_dict({"ACESTEP_LM_MODEL": DEFAULT_LM_MODEL})]
)
@modal.asgi_app()
def fastapi_app():
    # Import the already defined app creation function
    from acestep.api_server import create_app
    
    # Create and return the ASGI app for modal
    # Any necessary initialization (like lifespan) will be properly handled
    app = create_app()
    return app
