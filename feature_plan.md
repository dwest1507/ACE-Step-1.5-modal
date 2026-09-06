# Modal Deployment Implementation Spec

This document details the requirements, technical implementation plan, and the task checklist for integrating Modal serverless GPU support into the ACE-Step 1.5 project.

## 1. Requirements
- **Serverless Deployment**: Enable the ACE-Step 1.5 model to be deployed on [Modal](https://modal.com/) allowing for scale-to-zero serverless execution.
- **REST API Endpoint**: The deployment must expose a raw REST API endpoint so that it can be consumed by future custom frontends. It should not expose the Gradio UI.
- **Configurable Models**: The deployment script should default to using the `1.7B` LM model (a good balance of speed and VRAM usage on Modal GPUs like A10G or L4) but must allow the user to easily configure/override this choice.
- **Documentation**: Provide clear, step-by-step instructions for users to deploy this on their own Modal accounts.

## 2. Technical Implementation Plan

We will create a specific script (`modal_app.py`) designed to run within Modal's infrastructure, along with documentation.

### Core Architecture (`modal_app.py`)
- **Image Definition**: We will define a `modal.Image.debian_slim()` that installs Python dependencies (`uv`, `vllm`, `torch`, `acestep` requirements).
- **Model Volumes**: We will utilize a `modal.Volume` or `modal.sandbox` approach to download and cache the Hugging Face weights to prevent re-downloading them on every cold start.
- **App Configuration**: We will define a `modal.App("acestep-api")`.
- **API Definition**: We will use a `@app.cls()` with memory snapshotting (`@modal.enter(snap=True)`) to load models into GPU memory, and expose the ASGI app with FastAPI via `@modal.asgi_app()`. The API will mirror the existing `acestep-api` functionality to handle text-to-music generation requests.
- **Parameterization**: We will use environment variables from .env or .env.example to allow users to specify which LM model they want to load (e.g., `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B` by default, but overrideable). The appropriate GPU will be selected based on the model size. In addition the approapriate, model download command will be selected based on the model size.

### Affected/New Files

#### [NEW] `modal_app.py`
The main entry point for Modal. It will contain:
1. Docker image setup.
2. Build step to download models.
3. The Modal app class and memory snapshot setup.

#### [NEW] `docs/en/MODAL_GUIDE.md`
A Markdown guide detailing:
1. Setting up a Modal account and installing the CLI.
2. Configuring secrets/environment variables to change the default LM model.
3. Commands to deploy (`modal deploy modal_app.py`).
4. Example `curl` requests to interact with the raw REST API.

#### [MODIFY] [README.md](file:///home/david/Projects/open-source/ACE-Step-1.5/README.md)
Add a section linking to the new Modal deployment guide under the launch options.

#### [MODIFY] `acestep/api_server.py`
Added `set_preloaded_models()` function and module-level globals (`_preloaded_handler`, `_preloaded_llm`) to support pre-loaded model injection from Modal. These are passed through to the refactored helper modules below.

> [!NOTE]
> Upstream refactored `api_server.py` from a monolithic file into a thin orchestrator that delegates to helper modules under `acestep/api/`. The preloaded-model support was adapted to this new architecture.

#### [MODIFY] `acestep/api/lifespan_runtime.py`
Added optional `preloaded_handler` and `preloaded_llm` parameters to `initialize_lifespan_runtime()`. When provided, these pre-initialized handlers are used instead of creating new instances.

#### [MODIFY] `acestep/api/startup_model_init.py`
Added optional `preloaded_handler` and `preloaded_llm` parameters to `initialize_models_at_startup()`. In `no_init` mode, preloaded models are recognized as already-initialized.

## 3. Task Checklist

- [x] **Phase 1: Modal Script Creation**
  - [x] Create `modal_app.py`.
  - [x] Define the Modal Image and dependency installation.
  - [x] Implement the model caching/downloading logic.
  - [x] Implement the FastAPI/REST endpoint structure.
  - [x] Add environment variable support for changing the LM model (defaulting to 1.7B).
- [x] **Phase 2: Documentation**
  - [x] Create `docs/en/MODAL_GUIDE.md`.
  - [x] Write deployment instructions and API usage examples.
  - [x] Update [README.md](file:///home/david/Projects/open-source/ACE-Step-1.5/README.md) to reference the new guide.
- [ ] **Phase 3: Testing & Verification**
  - [x] Run `modal shell` or `modal serve` locally to test image build.
  - [x] Verify the API endpoint responds correctly to requests.
  - [ ] Final code review before opening the Pull Request.
- [x] **Phase 3.5: Upstream Merge Integration**
  - [x] Merge `main` into `feature/modal-support` and resolve conflicts.
  - [x] Adapt preloaded-model support to main's refactored `acestep/api/` module architecture.
  - [x] Verify existing unit tests pass with changes.
- [ ] **Phase 3.6: Second Upstream Merge (main @ `ca1e85f`)**
  - [x] Merge `main` into the feature branch and resolve conflicts in `pyproject.toml`,
        `acestep/audio_utils.py`, and `acestep/audio_utils_test.py`.
  - [x] Drop the fork's `_save_mp3` soundfile patch in favour of upstream's equivalent
        (upstream independently moved off `torchaudio.save` and added `detach().cpu()`
        plus a contiguity guard). The fork's `save_audio` FLAC/WAV/WAV32 patch is kept —
        upstream still routes those through `torchaudio.save(backend='soundfile')`, which
        torchaudio 2.10 delegates to torchcodec.
  - [x] Confirm the preloaded-model injection points survived upstream's API-layer churn
        (`initialize_lifespan_runtime`, `initialize_models_at_startup`,
        `ensure_models_initialized` fast path).
  - [x] Pin `ACESTEP_PROJECT_ROOT=/workspace` in the Modal image. Checkpoint resolution
        (`model_downloader.get_project_root`) falls back to `os.getcwd()`, and upstream
        added `ACESTEP_CHECKPOINTS_DIR` as a second override — both now resolve
        deterministically to the baked-in `/workspace/checkpoints`.
  - [x] Bake `ACESTEP_CONFIG_PATH` / `ACESTEP_LM_MODEL_PATH` / `ACESTEP_LM_BACKEND=pt`
        into the image env so a Secret that omits them cannot load different models —
        or, via `api/llm_readiness.py`, a different LM backend — than the image was
        built with.
  - [x] Verify no test regressions against an `origin/main` baseline: 977 passed, and
        the failure set is identical apart from two `release_task_audio_paths_test`
        cases that only fail on the baseline because that checkout lived under `/tmp`
        (the test asserts against the system temp dir). Every remaining failure
        reproduces on unmodified `main`.
  - [ ] **Human verification required:** run `uv run modal deploy modal_app.py` and
        exercise the deployed endpoint. No real deployment was made as part of this merge.
- [ ] **Phase 4: Future Improvements**
  - [ ] **nanovllm Memory Snapshot Compatibility** — The Modal deployment currently uses `backend="pt"` (PyTorch) for the LLM because nanovllm (`backend="vllm"`) contains CRIU-incompatible constructs (`threading.Lock`, `atexit.register`, `mp.get_context("spawn")`, CUDA graph capture) that prevent GPU memory snapshotting. The desired end state is to use the faster nanovllm backend with full snapshot support. Two paths forward:
    1. **Update nanovllm**: Refactor its `LLMEngine`/`ModelRunner` to defer CRIU-incompatible initialization (locks, atexit, multiprocessing, CUDA graphs) until after snapshot restore, or make them lazily initialized.
    2. **Work with Modal**: Investigate whether Modal can extend their CRIU-based snapshotting to handle these constructs (threading locks, atexit handlers), which would benefit any inference engine with similar patterns.
