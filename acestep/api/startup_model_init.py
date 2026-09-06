"""Startup model initialization orchestration for API server lifespan."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from acestep.gpu_config import (
    VRAM_AUTO_OFFLOAD_THRESHOLD_GB,
    get_gpu_config,
    set_global_gpu_config,
)
from acestep.api.startup_llm_init import initialize_llm_at_startup


def do_model_initialization(
    *,
    app: Any,
    handler: Any,
    llm_handler: Any,
    handler2: Any,
    handler3: Any,
    config_path2: str,
    config_path3: str,
    get_project_root: Callable[[], str],
    get_model_name: Callable[[str], str],
    ensure_model_downloaded: Callable[[str, str], str],
    env_bool: Callable[[str, bool], bool],
) -> None:
    """Download and load DiT, VAE, and optional LLM models.

    This is the actual model initialization logic, callable both at startup
    (when ``ACESTEP_NO_INIT=false``) and lazily on first request.
    """

    gpu_config = app.state.gpu_config
    gpu_memory_gb = gpu_config.gpu_memory_gb
    auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB

    print("[API Server] Initializing models...")
    if auto_offload:
        print("[API Server] Auto-enabling CPU offload (GPU < 16GB)")
    elif gpu_memory_gb > 0:
        print("[API Server] CPU offload disabled by default (GPU >= 16GB)")
    else:
        print("[API Server] No GPU detected, running on CPU")

    project_root = get_project_root()
    config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    device = os.getenv("ACESTEP_DEVICE", "auto")
    use_flash_attention = env_bool("ACESTEP_USE_FLASH_ATTENTION", True)

    offload_to_cpu_env = os.getenv("ACESTEP_OFFLOAD_TO_CPU")
    if offload_to_cpu_env is not None:
        offload_to_cpu = env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
    else:
        offload_to_cpu = auto_offload
        if auto_offload:
            print("[API Server] Auto-setting offload_to_cpu=True based on GPU memory")

    offload_dit_to_cpu = env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False)
    compile_model = env_bool("ACESTEP_COMPILE_MODEL", False)

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dit_model_name = get_model_name(config_path)
    if dit_model_name:
        try:
            ensure_model_downloaded(dit_model_name, checkpoint_dir)
        except Exception as exc:
            print(f"[API Server] Warning: Failed to download DiT model: {exc}")

    try:
        ensure_model_downloaded("vae", checkpoint_dir)
    except Exception as exc:
        print(f"[API Server] Warning: Failed to download VAE model: {exc}")

    print(f"[API Server] Loading primary DiT model: {config_path}")
    status_msg, ok = handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
    )
    if not ok:
        app.state._init_error = status_msg
        print(f"[API Server] ERROR: Primary model failed to load: {status_msg}")
        raise RuntimeError(status_msg)
    app.state._initialized = True
    # Captured for on-demand model switching (ACESTEP_ON_DEMAND_MODEL_LOAD):
    # a later request for an unloaded model re-runs initialize_service on the
    # primary handler with these same kwargs. Named distinctly from the
    # lazy-init kwargs some runtimes store as _model_init_kwargs, which have
    # an incompatible do_model_initialization(**kwargs) shape.
    app.state._service_init_kwargs = {
        "project_root": project_root,
        "device": device,
        "use_flash_attention": use_flash_attention,
        "compile_model": compile_model,
        "offload_to_cpu": offload_to_cpu,
        "offload_dit_to_cpu": offload_dit_to_cpu,
    }
    app.state._checkpoint_dir = checkpoint_dir
    app.state._ensure_model_downloaded = ensure_model_downloaded
    print(f"[API Server] Primary model loaded: {get_model_name(config_path)}")

    if handler2 and config_path2:
        model2_name = get_model_name(config_path2)
        if model2_name:
            try:
                ensure_model_downloaded(model2_name, checkpoint_dir)
            except Exception as exc:
                print(f"[API Server] Warning: Failed to download secondary model: {exc}")
        print(f"[API Server] Loading secondary DiT model: {config_path2}")
        try:
            status_msg2, ok2 = handler2.initialize_service(
                project_root=project_root,
                config_path=config_path2,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
            )
            app.state._initialized2 = ok2
            if ok2:
                print(f"[API Server] Secondary model loaded: {model2_name}")
            else:
                print(f"[API Server] Warning: Secondary model failed: {status_msg2}")
        except Exception as exc:
            print(f"[API Server] Warning: Failed to initialize secondary model: {exc}")
            app.state._initialized2 = False

    if handler3 and config_path3:
        model3_name = get_model_name(config_path3)
        if model3_name:
            try:
                ensure_model_downloaded(model3_name, checkpoint_dir)
            except Exception as exc:
                print(f"[API Server] Warning: Failed to download third model: {exc}")
        print(f"[API Server] Loading third DiT model: {config_path3}")
        try:
            status_msg3, ok3 = handler3.initialize_service(
                project_root=project_root,
                config_path=config_path3,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
            )
            app.state._initialized3 = ok3
            if ok3:
                print(f"[API Server] Third model loaded: {model3_name}")
            else:
                print(f"[API Server] Warning: Third model failed: {status_msg3}")
        except Exception as exc:
            print(f"[API Server] Warning: Failed to initialize third model: {exc}")
            app.state._initialized3 = False

    initialize_llm_at_startup(
        app=app,
        llm_handler=llm_handler,
        gpu_config=gpu_config,
        device=device,
        offload_to_cpu=offload_to_cpu,
        checkpoint_dir=checkpoint_dir,
        get_model_name=get_model_name,
        ensure_model_downloaded=ensure_model_downloaded,
        env_bool=env_bool,
    )

    print("[API Server] All models initialized successfully!")


def initialize_models_at_startup(
    *,
    app: Any,
    handler: Any,
    llm_handler: Any,
    handler2: Any,
    handler3: Any,
    config_path2: str,
    config_path3: str,
    get_project_root: Callable[[], str],
    get_model_name: Callable[[str], str],
    ensure_model_downloaded: Callable[[str, str], str],
    env_bool: Callable[[str, bool], bool],
    preloaded_handler: Optional[Any] = None,
    preloaded_llm: Optional[Any] = None,
) -> None:
    """Detect GPU configuration and optionally initialize models at startup.

    By default models are NOT loaded at startup (lazy-loaded on first request).
    Set ``ACESTEP_NO_INIT=false`` to force eager loading at startup.
    """

    no_init = env_bool("ACESTEP_NO_INIT", True)
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    app.state.gpu_config = gpu_config

    gpu_memory_gb = gpu_config.gpu_memory_gb

    print(f"\n{'='*60}")
    print("[API Server] GPU Configuration Detected:")
    print(f"{'='*60}")
    print(f"  GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"  Configuration Tier: {gpu_config.tier}")
    print(f"  Max Duration (with LM): {gpu_config.max_duration_with_lm}s")
    print(f"  Max Duration (without LM): {gpu_config.max_duration_without_lm}s")
    print(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    print(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    print(f"  Default LM Init: {gpu_config.init_lm_default}")
    print(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    print(f"{'='*60}\n")

    # Store init kwargs on app.state so lazy initialization can reuse them
    app.state._model_init_kwargs = dict(
        handler=handler,
        llm_handler=llm_handler,
        handler2=handler2,
        handler3=handler3,
        config_path2=config_path2,
        config_path3=config_path3,
        get_project_root=get_project_root,
        get_model_name=get_model_name,
        ensure_model_downloaded=ensure_model_downloaded,
        env_bool=env_bool,
    )

    # Pre-loaded model injection (e.g. from Modal GPU memory snapshot).
    # Setting _initialized=True causes ensure_models_initialized() to fast-path,
    # so the lazy-load path never fires.
    #
    # app.state._service_init_kwargs is deliberately left unset here: it only
    # drives on-demand model switching (ACESTEP_ON_DEMAND_MODEL_LOAD), which
    # would swap the snapshot-resident model for one fetched at request time.
    # job_model_selection treats missing kwargs as "fall back to the primary
    # model and log", which is the correct outcome for a snapshot deployment
    # that ships exactly one DiT model in its image.
    if preloaded_handler is not None:
        app.state._initialized = True
        if preloaded_llm and getattr(preloaded_llm, 'llm_initialized', False):
            app.state._llm_initialized = True
        else:
            # The LM did not come back with the snapshot, and a restored container
            # must not re-initialize it at request time. Both lazy paths would pick
            # the backend themselves: llm_readiness resolves
            # `req.lm_backend or os.getenv("ACESTEP_LM_BACKEND") or "vllm"`, and
            # GenerateMusicRequest.lm_backend defaults to the string "vllm" rather
            # than None -- so the request always wins and the deployment's
            # ACESTEP_LM_BACKEND=pt is never consulted. That would load nanovllm,
            # whose CRIU-incompatible internals are the reason this deployment pins
            # the PyTorch backend in the first place.
            #
            # Recording the failure instead makes the outcome deterministic and
            # self-explanatory: optional LLM features (use_cot_caption,
            # use_cot_language) auto-disable, requests that genuinely require the
            # LM fail with the message below, and /create_random_sample and
            # /format_input surface it rather than starting a multi-GB download
            # inside a request. Both flags are set because the two readers differ
            # -- llm_readiness short-circuits on _llm_init_error, while
            # model_init_service clears _llm_lazy_load_disabled when an operator
            # re-initializes the LM deliberately through /models.
            app.state._llm_initialized = False
            app.state._llm_lazy_load_disabled = True
            app.state._llm_init_error = (
                "LLM was not restored with the GPU snapshot. Request-time LLM "
                "initialization is disabled for snapshot deployments because it "
                "would load a different backend than the image was built with. "
                "Redeploy so the snapshot captures the LM, or run without "
                "pre-loaded models."
            )
            print("[API Server] Pre-loaded LLM unavailable; lazy LLM init disabled")
        print("[API Server] Using pre-loaded models (external entry point / Modal snapshot)")
        return

    if no_init:
        print("[API Server] Models will be lazy-loaded on first request")
        print("[API Server] Set ACESTEP_NO_INIT=false to load models at startup")
        print("[API Server] Server is ready to accept requests (models not loaded yet)")
        return

    print("[API Server] Eager model loading enabled (ACESTEP_NO_INIT=false)")
    do_model_initialization(
        app=app,
        **app.state._model_init_kwargs,
    )
