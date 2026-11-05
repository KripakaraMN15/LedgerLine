CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    goal TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'created',
            'running',
            'awaiting_approval',
            'done',
            'failed',
            'killed',
            'cancelled'
        )
    ),
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_tasks_status_updated_at
    ON tasks (status, updated_at);

CREATE TABLE IF NOT EXISTS task_events (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    from_status TEXT NOT NULL CHECK (
        from_status IN (
            'created',
            'running',
            'awaiting_approval',
            'done',
            'failed',
            'killed',
            'cancelled'
        )
    ),
    to_status TEXT NOT NULL CHECK (
        to_status IN (
            'created',
            'running',
            'awaiting_approval',
            'done',
            'failed',
            'killed',
            'cancelled'
        )
    ),
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_task_events_task_id_created_at
    ON task_events (task_id, created_at, id);
