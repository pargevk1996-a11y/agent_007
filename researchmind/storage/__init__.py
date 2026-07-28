"""PostgreSQL access: connection pool, hand-written SQL repositories, migration runner.

Sole owner of transactions; no module above this one opens one. Sole place where JSONB
is deserialised into typed models. Every query is scoped by tenant_id.
"""
