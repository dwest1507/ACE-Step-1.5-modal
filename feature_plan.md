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
- **API Definition**: We will use `@app.function()` with a web endpoint (`@modal.web_endpoint` or ASGI app with FastAPI). The API will mirror the existing `acestep-api` functionality to handle text-to-music generation requests.
- **Parameterization**: We will use Modal's Secrets or environment variables to allow users to specify which LM model they want to load (e.g., `ACESTEP_LM_MODEL=1.7B` by default, but overrideable).

### Affected/New Files

#### [NEW] `modal_app.py`
The main entry point for Modal. It will contain:
1. Docker image setup.
2. Build step to download models.
3. The Modal app and web API setup.

#### [NEW] `docs/en/MODAL_GUIDE.md`
A Markdown guide detailing:
1. Setting up a Modal account and installing the CLI.
2. Configuring secrets/environment variables to change the default LM model.
3. Commands to deploy (`modal deploy modal_app.py`).
4. Example `curl` requests to interact with the raw REST API.

#### [MODIFY] [README.md](file:///home/david/Projects/open-source/ACE-Step-1.5/README.md)
Add a section linking to the new Modal deployment guide under the launch options.

## 3. Task Checklist

- [ ] **Phase 1: Modal Script Creation**
  - [ ] Create `modal_app.py`.
  - [ ] Define the Modal Image and dependency installation.
  - [ ] Implement the model caching/downloading logic.
  - [ ] Implement the FastAPI/REST endpoint structure.
  - [ ] Add environment variable support for changing the LM model (defaulting to 1.7B).
- [ ] **Phase 2: Documentation**
  - [ ] Create `docs/en/MODAL_GUIDE.md`.
  - [ ] Write deployment instructions and API usage examples.
  - [ ] Update [README.md](file:///home/david/Projects/open-source/ACE-Step-1.5/README.md) to reference the new guide.
- [ ] **Phase 3: Testing & Verification**
  - [ ] Run `modal shell` or `modal serve` locally to test image build.
  - [ ] Verify the API endpoint responds correctly to requests.
  - [ ] Final code review before opening the Pull Request.
