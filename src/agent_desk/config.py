"""One versioned configuration (configs/app.yaml) plus experiment overrides.

    cfg = get_config()                      # app.yaml (+ AGENT_DESK_EXPERIMENT overrides)
    cfg["llm"]["models"]["lead"]
    with_overrides({"critic": {"enabled": False}})   # a new dict; the cached one is untouched

Environment variables:
    AGENT_DESK_EXPERIMENT  path to an experiment YAML whose keys override app.yaml
    AGENT_DESK_LLM_MODE    live | offline (overrides llm.mode)
    AGENT_DESK_DATA        data directory (default <project>/data)
The .env file is looked up in the project root and then in parent folders, so a key file kept
one level up (outside the repository) is also found. Its values are never logged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"


def _load_env() -> None:
    # Nearest file first; override=False means a key set closer to the project (or in the real
    # environment) is never replaced by one further up.
    for folder in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        env_file = folder / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)


_load_env()


def data_dir() -> Path:
    path = Path(os.getenv("AGENT_DESK_DATA", PROJECT_ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def _base_config() -> dict:
    cfg = load_yaml(CONFIG_PATH)
    experiment = os.getenv("AGENT_DESK_EXPERIMENT")
    if experiment:
        override = load_yaml(experiment)
        override.pop("name", None)
        override.pop("description", None)
        cfg = deep_merge(cfg, override)
    mode = os.getenv("AGENT_DESK_LLM_MODE")
    if mode:
        cfg["llm"]["mode"] = mode
    return cfg


def get_config() -> dict:
    """A copy of the active configuration (safe to mutate)."""
    return copy.deepcopy(_base_config())


def with_overrides(override: dict | None) -> dict:
    return deep_merge(_base_config(), override or {})


def reset_config_cache() -> None:
    _base_config.cache_clear()


def config_fingerprint(cfg: dict) -> str:
    """Short hash of the effective config, recorded with every run so results are attributable."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def needs_api_key(cfg: dict) -> bool:
    """A hosted provider needs OPENROUTER_API_KEY; a local OpenAI-compatible server (Ollama,
    LM Studio, llama.cpp) at localhost does not."""
    base = cfg["llm"].get("base_url", "")
    return not any(host in base for host in ("localhost", "127.0.0.1", "host.docker.internal"))


def api_key(cfg: dict) -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key if needs_api_key(cfg) else (os.getenv("LOCAL_LLM_API_KEY", "") or "local")


def is_offline(cfg: dict | None = None) -> bool:
    cfg = cfg or _base_config()
    if cfg["llm"]["mode"] == "offline":
        return True
    return needs_api_key(cfg) and not os.getenv("OPENROUTER_API_KEY", "").strip()


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
