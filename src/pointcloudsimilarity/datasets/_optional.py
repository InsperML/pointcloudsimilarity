import importlib


def import_optional(module, needed_by, extra='data'):
    """
    Import an optional dependency, or explain how to install it.

    The loaders import their dependencies only when called, so importing
    pointcloudsimilarity.datasets stays cheap and works without them.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{needed_by} needs the '{module}' package. Install it with "
            f'`uv sync --extra {extra}` (or `pip install pointcloudsimilarity[{extra}]`).'
        ) from e
