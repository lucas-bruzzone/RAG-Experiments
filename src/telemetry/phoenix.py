"""Phoenix telemetry integration"""

import os

try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("Warning: Phoenix not installed. Run: pip install arize-phoenix openinference-instrumentation-langchain")


class PhoenixTelemetry:
    """Manages Phoenix telemetry lifecycle"""

    _session = None
    _tracer_provider = None

    @classmethod
    def start(cls, launch_ui: bool = True, project_name: str = "rag-app"):
        """Initialize Phoenix telemetry"""
        if not PHOENIX_AVAILABLE:
            print("Phoenix not available - continuing without telemetry")
            return

        if cls._session is None:
            os.environ["PHOENIX_VERBOSE"] = "false"

            # Just register tracer, don't launch UI (run Phoenix separately)
            cls._tracer_provider = register(
                project_name=project_name,
                auto_instrument=True
            )
            print("✓ LangChain auto-instrumentation enabled")
            print("📊 Make sure Phoenix is running: python -m phoenix.server.main serve")
            print("🌐 Then visit: http://localhost:6006")

    @classmethod
    def stop(cls):
        """Stop Phoenix session"""
        if cls._session is not None:
            # Phoenix 12.x ThreadSession doesn't need explicit close
            cls._session = None
            cls._tracer_provider = None
            print("Phoenix session ended (UI still running in background)")

    @classmethod
    def is_running(cls):
        """Check if Phoenix is active"""
        return cls._session is not None
