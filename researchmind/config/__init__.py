"""Environment-derived settings.

The single place allowed to read os.environ. Every other package receives an
already-validated settings object rather than reaching for the environment itself.
"""
