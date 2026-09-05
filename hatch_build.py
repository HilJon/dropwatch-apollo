"""Mark the bundled vendor DLL wheel as Windows x86-64 specific."""

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            # Source development/replay is portable; live DLLs remain Windows-only.
            return
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-win_amd64"
