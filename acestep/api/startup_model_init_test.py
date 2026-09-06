"""Unit tests for startup model initialization orchestration helper."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from acestep.api.startup_model_init import initialize_models_at_startup


def _gpu_config(init_lm_default: bool = True) -> SimpleNamespace:
    """Create a fake GPU config object with all fields used by startup init."""

    return SimpleNamespace(
        gpu_memory_gb=24.0,
        tier="high",
        max_duration_with_lm=180,
        max_duration_without_lm=300,
        max_batch_size_with_lm=2,
        max_batch_size_without_lm=4,
        init_lm_default=init_lm_default,
        available_lm_models=["acestep-5Hz-lm-1.1B"],
    )


class StartupModelInitTests(unittest.TestCase):
    """Behavior tests for startup model-loading orchestration."""

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_skips_model_init_in_no_init_mode(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should skip model initialization when ACESTEP_NO_INIT resolves true."""

        app = SimpleNamespace(state=SimpleNamespace())
        handler = MagicMock()
        llm_handler = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            return True if name == "ACESTEP_NO_INIT" else default

        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=None,
            handler3=None,
            config_path2="",
            config_path3="",
            get_project_root=MagicMock(return_value="k:/repo"),
            get_model_name=MagicMock(return_value="acestep-v15-turbo"),
            ensure_model_downloaded=MagicMock(),
            env_bool=_env_bool,
        )

        handler.initialize_service.assert_not_called()
        mock_initialize_llm_at_startup.assert_not_called()
        self.assertIsNotNone(getattr(app.state, "gpu_config", None))

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_default_lazy_mode_skips_loading(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Default behavior should skip model loading (lazy-load on first request)."""

        app = SimpleNamespace(state=SimpleNamespace())
        handler = MagicMock()
        llm_handler = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        # Use default env_bool — ACESTEP_NO_INIT defaults to True (lazy mode)
        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=None,
            handler3=None,
            config_path2="",
            config_path3="",
            get_project_root=MagicMock(return_value="k:/repo"),
            get_model_name=MagicMock(return_value="acestep-v15-turbo"),
            ensure_model_downloaded=MagicMock(),
            env_bool=lambda _name, default: default,
        )

        handler.initialize_service.assert_not_called()
        mock_initialize_llm_at_startup.assert_not_called()
        self.assertIsNotNone(getattr(app.state, "gpu_config", None))
        # Init kwargs should be stored for lazy use
        self.assertIsNotNone(getattr(app.state, "_model_init_kwargs", None))

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_initializes_primary_and_calls_llm(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should initialize primary DiT and then call LLM startup helper
        when ACESTEP_NO_INIT is explicitly set to False."""

        app = SimpleNamespace(
            state=SimpleNamespace(
                _initialized=False,
                _initialized2=False,
                _initialized3=False,
                _init_error=None,
                _llm_initialized=False,
                _llm_init_error=None,
                _llm_lazy_load_disabled=False,
            )
        )
        handler = MagicMock()
        handler.initialize_service.return_value = ("ok", True)
        llm_handler = MagicMock()
        ensure_model_downloaded = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            # Force ACESTEP_NO_INIT=False to trigger eager loading
            return False if name == "ACESTEP_NO_INIT" else default

        with patch.dict(os.environ, {}, clear=True):
            initialize_models_at_startup(
                app=app,
                handler=handler,
                llm_handler=llm_handler,
                handler2=None,
                handler3=None,
                config_path2="",
                config_path3="",
                get_project_root=MagicMock(return_value="k:/repo"),
                get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                ensure_model_downloaded=ensure_model_downloaded,
                env_bool=_env_bool,
            )

        handler.initialize_service.assert_called_once()
        ensure_model_downloaded.assert_any_call("acestep-v15-turbo", ANY)
        ensure_model_downloaded.assert_any_call("vae", ANY)
        self.assertTrue(app.state._initialized)
        mock_initialize_llm_at_startup.assert_called_once()

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_raises_on_primary_init_failure(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should raise and persist init error when primary DiT fails."""

        app = SimpleNamespace(
            state=SimpleNamespace(
                _initialized=False,
                _init_error=None,
                _llm_initialized=False,
                _llm_init_error=None,
                _llm_lazy_load_disabled=False,
            )
        )
        handler = MagicMock()
        handler.initialize_service.return_value = ("boom", False)
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            # Force ACESTEP_NO_INIT=False to trigger eager loading
            return False if name == "ACESTEP_NO_INIT" else default

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                initialize_models_at_startup(
                    app=app,
                    handler=handler,
                    llm_handler=MagicMock(),
                    handler2=None,
                    handler3=None,
                    config_path2="",
                    config_path3="",
                    get_project_root=MagicMock(return_value="k:/repo"),
                    get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                    ensure_model_downloaded=MagicMock(),
                    env_bool=_env_bool,
                )

        mock_initialize_llm_at_startup.assert_not_called()
        self.assertEqual("boom", app.state._init_error)


class PreloadedModelInjectionTests(unittest.TestCase):
    """Behavior tests for the pre-loaded (Modal GPU snapshot) injection path."""

    def _initialize(self, app, preloaded_handler, preloaded_llm):
        """Run startup init in preloaded mode with the GPU probe stubbed out."""

        with (
            patch("acestep.api.startup_model_init.initialize_llm_at_startup") as mock_llm_startup,
            patch("acestep.api.startup_model_init.set_global_gpu_config"),
            patch("acestep.api.startup_model_init.get_gpu_config", return_value=_gpu_config()),
        ):
            initialize_models_at_startup(
                app=app,
                handler=MagicMock(),
                llm_handler=MagicMock(),
                handler2=None,
                handler3=None,
                config_path2="",
                config_path3="",
                get_project_root=MagicMock(return_value="/workspace"),
                get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                ensure_model_downloaded=MagicMock(),
                env_bool=lambda _name, default: default,
                preloaded_handler=preloaded_handler,
                preloaded_llm=preloaded_llm,
            )
        return mock_llm_startup

    def test_preloaded_handler_marks_models_initialized(self) -> None:
        """A snapshot-resident DiT model must not be re-initialized on first request."""

        app = SimpleNamespace(state=SimpleNamespace())

        mock_llm_startup = self._initialize(
            app,
            preloaded_handler=MagicMock(),
            preloaded_llm=SimpleNamespace(llm_initialized=True),
        )

        self.assertTrue(app.state._initialized)
        mock_llm_startup.assert_not_called()

    def test_preloaded_llm_that_loaded_is_marked_available(self) -> None:
        """An LM restored with the snapshot stays usable without a reload."""

        app = SimpleNamespace(state=SimpleNamespace())

        self._initialize(
            app,
            preloaded_handler=MagicMock(),
            preloaded_llm=SimpleNamespace(llm_initialized=True),
        )

        self.assertTrue(app.state._llm_initialized)
        self.assertFalse(getattr(app.state, "_llm_lazy_load_disabled", False))
        self.assertIsNone(getattr(app.state, "_llm_init_error", None))

    def test_preloaded_llm_that_failed_blocks_request_time_reload(self) -> None:
        """A snapshot deployment must not re-init the LM inside a request.

        llm_readiness would resolve the backend from the request, whose
        lm_backend field defaults to "vllm" rather than None -- so the
        deployment's ACESTEP_LM_BACKEND=pt never applies and a CRIU-incompatible
        nanovllm load would start on a restored container. Reporting the LM as
        unavailable keeps the outcome deterministic instead.
        """

        app = SimpleNamespace(state=SimpleNamespace())

        self._initialize(
            app,
            preloaded_handler=MagicMock(),
            preloaded_llm=SimpleNamespace(llm_initialized=False),
        )

        self.assertTrue(app.state._initialized)
        self.assertFalse(app.state._llm_initialized)
        self.assertTrue(app.state._llm_lazy_load_disabled)
        self.assertIn("snapshot", app.state._llm_init_error)

    def test_missing_preloaded_llm_blocks_request_time_reload(self) -> None:
        """An entry point that injects only a DiT handler gets the same guard."""

        app = SimpleNamespace(state=SimpleNamespace())

        self._initialize(app, preloaded_handler=MagicMock(), preloaded_llm=None)

        self.assertTrue(app.state._llm_lazy_load_disabled)
        self.assertIsNotNone(app.state._llm_init_error)


if __name__ == "__main__":
    unittest.main()
