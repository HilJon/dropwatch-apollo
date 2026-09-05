"""Public Dropwatch Apollo API."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("dropwatch-apollo")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from dropwatch.apollo import DropwatchApollo
from dropwatch.models import ApolloEvaluationError
from dropwatch.models import ApolloFrameLossError
from dropwatch.models import ApolloFrameSource
from dropwatch.models import ApolloIncompleteSequenceError
from dropwatch.models import ApolloLifecycleError
from dropwatch.models import ApolloSequenceEvaluator
from dropwatch.models import ApolloSettings
from dropwatch.models import ApolloStats
from dropwatch.models import ApolloTransportError
from dropwatch.models import ApolloVideoSettings
from dropwatch.replay import ReplayFrameSource

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
    "ReplayFrameSource",
    "__version__",
]
