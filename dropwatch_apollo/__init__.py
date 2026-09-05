"""Public Dropwatch Apollo API."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("dropwatch-apollo")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from dropwatch_apollo.apollo import DropwatchApollo
from dropwatch_apollo.models import ApolloEvaluationError
from dropwatch_apollo.models import ApolloFrameLossError
from dropwatch_apollo.models import ApolloFrameSource
from dropwatch_apollo.models import ApolloIncompleteSequenceError
from dropwatch_apollo.models import ApolloLifecycleError
from dropwatch_apollo.models import ApolloSequenceEvaluator
from dropwatch_apollo.models import ApolloSettings
from dropwatch_apollo.models import ApolloStats
from dropwatch_apollo.models import ApolloTransportError
from dropwatch_apollo.models import ApolloVideoSettings
from dropwatch_apollo.models import DisplayRoi2D
from dropwatch_apollo.replay import ReplayFrameSource

__all__ = [
    "ApolloFrameSource",
    "ApolloFrameLossError",
    "ApolloIncompleteSequenceError",
    "ApolloEvaluationError",
    "ApolloLifecycleError",
    "ApolloSequenceEvaluator",
    "ApolloSettings",
    "ApolloStats",
    "ApolloTransportError",
    "ApolloVideoSettings",
    "DropwatchApollo",
    "DisplayRoi2D",
    "ReplayFrameSource",
    "__version__",
]
