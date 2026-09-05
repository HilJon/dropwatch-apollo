"""Compatibility imports provided by dropwatch-apollo, not the retired recorder engine."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    version("dropwatch-recorder")
except PackageNotFoundError:
    pass
else:
    raise ImportError(
        "dropwatch-recorder and dropwatch-apollo both provide 'recorder'. "
        "Uninstall dropwatch-recorder, then reinstall dropwatch-apollo in this environment."
    )
