"""LLM and embedding provider interfaces plus vendor adapters.

The only package allowed to import vendor SDKs. Core logic depends on the interface
and never on a vendor, which is what makes the provider a configuration choice.
"""
