"""Typed tool contracts and their implementations.

Every tool declares a Pydantic input schema, a Pydantic output schema, a cost model,
a timeout and a failure taxonomy. The model sees the schemas; the runtime enforces them.
"""
