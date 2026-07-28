"""Long-lived process that consumes run messages and drives the agent.

Orchestrates planner, executor, critic and synthesizer as typed Python calls rather
than network hops. Restarting it must never lose a run.
"""
