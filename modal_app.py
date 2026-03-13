import os
from pathlib import Path
import modal

# Load .env so config is available locally during `modal deploy`
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent / ".env.example"
    load_dotenv(env_path, override=False)
except ImportError:
    pass  # dotenv optional; user can set env vars directly

# Create a Modal app
app = modal.App("acestep-api")

# Get defaults from environment variables.
# Note: these are read locally at `modal deploy` time to configure the image/container.
# At runtime, the Modal Secret "ace-step-api-secrets" re-populates these inside the container.
LM_MODEL_PATH = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

# Definition of the models to download during image build
def download_models():
    import os
    from huggingface_hub import snapshot_download
    
    print("Downloading ACE-Step 1.5 core models...")
    # The default repo
    snapshot_download(
        repo_id="ACE-Step/Ace-Step1.5", 
        local_dir="/workspace/checkpoints"
    )

    # Optional: download other models if ACESTEP_LM_MODEL_PATH is customized
    lm_path = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
    if "0.6B" in lm_path:
        print("Downloading ACE-Step-5Hz-lm-0.6B...")
        snapshot_download(
            repo_id="ACE-Step/acestep-5Hz-lm-0.6B",
            local_dir="/workspace/checkpoints/acestep-5Hz-lm-0.6B"
        )
    elif "4B" in lm_path:
        print("Downloading ACE-Step-5Hz-lm-4B...")
        snapshot_download(
            repo_id="ACE-Step/acestep-5Hz-lm-4B",
            local_dir="/workspace/checkpoints/acestep-5Hz-lm-4B"
        )

def _modal_gpu_string() -> str:
    """Return the optimal Modal GPU string based on the configured LM model.

    This intentionally uses only string-matching against LM_MODEL_PATH and avoids
    importing torch or any GPU-probing code. `modal deploy` runs this function on
    the *local* machine (which may be CPU-only), so any torch import here would
    break deployment from dev machines.

    GPU ↔ model mapping:
      0.6B → L4   (24 GB, cost-effective for the smallest model)
      1.7B → A10G (24 GB, good balance of speed and cost)
      4B   → A100 (40/80 GB, required for the largest model)
    """
    lm = LM_MODEL_PATH.upper()
    if "4B" in lm:
        return "A100"
    elif "1.7B" in lm:
        return "A10G"
    else:
        return "L4"

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
        secrets=[
            modal.Secret.from_name("ace-step-api-secrets")
        ]
    )
)

@app.cls(
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    image=image,
    gpu=_modal_gpu_string(),  # resolved from LM_MODEL_PATH at deploy time (string-only, no torch)
    scaledown_window=300,     # Keep the container warm for 5 minutes after last request
    timeout=3600,
    secrets=[
        modal.Secret.from_name("ace-step-api-secrets")
    ]
)
class AceStepAPI:
    @modal.enter(snap=True)
    def load_models(self):
        """Load models directly into GPU memory before the snapshot is taken."""
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.gpu_config import get_gpu_config, set_global_gpu_config

        print("[Modal] Initializing models for GPU memory snapshot...")

        # Detect and set GPU config
        gpu_config = get_gpu_config()
        set_global_gpu_config(gpu_config)
        print(f"[Modal] GPU: {gpu_config.gpu_memory_gb:.1f} GB, tier: {gpu_config.tier}")

        # Load primary DiT model
        config_path = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
        handler = AceStepHandler()
        status, ok = handler.initialize_service(
            project_root="/workspace",
            config_path=config_path,
            device="auto",
            use_flash_attention=True,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"[Modal] DiT model failed to load: {status}")
        self._handler = handler
        print(f"[Modal] Primary DiT model loaded: {config_path}")

        # Load LLM
        lm_model = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
        llm = LLMHandler()
        status, ok = llm.initialize(
            checkpoint_dir="/workspace/checkpoints",
            lm_model_path=lm_model,
            backend="pt",  # Use PyTorch backend; nanovllm ("vllm") has CRIU-incompatible internals required for memory snapshotting (faster cold starts)
            device="auto",
            offload_to_cpu=False,
            dtype=None,
        )
        # Always store the handler reference regardless of whether initialization succeeded.
        # api_server.set_preloaded_models() accepts it unconditionally, and the lifespan
        # checks `getattr(_preloaded_llm, 'llm_initialized', False)` before enabling LLM
        # features — so passing a partially-initialized handler here is safe and intentional.
        self._llm_handler = llm
        if ok:
            print(f"[Modal] LLM model loaded: {lm_model}")
        else:
            print(f"[Modal] LLM model failed to load (non-fatal): {status}")

        print("[Modal] All models loaded — ready for GPU memory snapshot")

    @modal.asgi_app()
    def web(self):
        """Create the FastAPI app with pre-loaded models injected."""
        # Tell the lifespan to skip model initialization
        os.environ["ACESTEP_NO_INIT"] = "1"

        from acestep.api_server import create_app, set_preloaded_models

        # Pass pre-loaded model references to the lifespan
        set_preloaded_models(self._handler, self._llm_handler)

        return create_app()
