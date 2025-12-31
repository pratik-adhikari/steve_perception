"""Helpers for resolving config file paths.

Launch files often pass either:
  - an absolute path to a YAML/INI file, or
  - a short file name living in `share/<pkg>/config/`.

Nodes should accept both.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def resolve_pkg_config_path(package: str, cfg: str) -> Path:
    """Resolve a config path passed via ROS parameters.

    Args:
        package: ROS 2 package name (for share directory).
        cfg: Path string or file name.

    Returns:
        Absolute Path to an existing file.

    Raises:
        FileNotFoundError if the resolved path does not exist.
    """
    p = Path(str(cfg)).expanduser()

    # 1) Absolute/relative path that already exists.
    if p.is_file():
        return p.resolve()

    # 2) Treat as a file under <pkg_share>/config/
    pkg_share = Path(get_package_share_directory(package))
    candidate = (pkg_share / "config" / p).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Config file not found. Got cfg='{cfg}'. Tried: '{p}' and '{candidate}'."
    )
