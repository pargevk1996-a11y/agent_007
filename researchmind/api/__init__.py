"""FastAPI control plane: run lifecycle, plan approval, SSE event stream, cancellation.

Short-lived request and response process. Holds no run state in memory: a run awaiting
human approval waits in the database, not in a coroutine.
"""
