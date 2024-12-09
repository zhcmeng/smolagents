
from transformers.utils.import_utils import _is_package_available

_pygments_available = _is_package_available("pygments")

def is_pygments_available():
    return _pygments_available

from rich.console import Console
console = Console()
