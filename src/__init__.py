"""Violence judicial files classification package."""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


# Load local environment variables once at package import time.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
