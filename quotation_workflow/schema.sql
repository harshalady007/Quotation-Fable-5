PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workflow_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('estimator', 'approver', 'admin')),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version >= 1)
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_code TEXT NOT NULL COLLATE NOCASE UNIQUE,
    name TEXT NOT NULL,
    client_name TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (length(currency) = 3),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    created_by TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version >= 1)
);

CREATE TABLE IF NOT EXISTS quotations (
    quotation_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    quotation_number TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('draft', 'submitted', 'approved', 'rejected')
    ),
    current_revision INTEGER NOT NULL DEFAULT 1 CHECK (current_revision >= 1),
    created_by TEXT NOT NULL REFERENCES users(user_id),
    submitted_by TEXT REFERENCES users(user_id),
    approved_by TEXT REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version >= 1),
    UNIQUE(project_id, quotation_number)
);

CREATE TABLE IF NOT EXISTS quotation_revisions (
    revision_id TEXT PRIMARY KEY,
    quotation_id TEXT NOT NULL REFERENCES quotations(quotation_id),
    revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
    status TEXT NOT NULL CHECK (
        status IN ('draft', 'submitted', 'approved', 'rejected')
    ),
    notes TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL CHECK (length(currency) = 3),
    subtotal TEXT NOT NULL DEFAULT '0.00',
    item_digest TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES users(user_id),
    submitted_by TEXT REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    locked_at TEXT,
    UNIQUE(quotation_id, revision_number)
);

CREATE TABLE IF NOT EXISTS quotation_items (
    item_id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL REFERENCES quotation_revisions(revision_id),
    line_number INTEGER NOT NULL CHECK (line_number >= 1),
    description TEXT NOT NULL,
    unit TEXT NOT NULL,
    quantity TEXT NOT NULL,
    unit_price TEXT NOT NULL,
    line_total TEXT NOT NULL,
    pricing_snapshot_json TEXT,
    UNIQUE(revision_id, line_number)
);

CREATE TABLE IF NOT EXISTS approval_decisions (
    decision_id TEXT PRIMARY KEY,
    quotation_id TEXT NOT NULL REFERENCES quotations(quotation_id),
    revision_id TEXT NOT NULL REFERENCES quotation_revisions(revision_id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    comment TEXT NOT NULL DEFAULT '',
    actor_user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    UNIQUE(revision_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor_user_id TEXT NOT NULL REFERENCES users(user_id),
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    actor_user_id TEXT NOT NULL REFERENCES users(user_id),
    request_id TEXT NOT NULL,
    command_name TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(actor_user_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_quotations_project
    ON quotations(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_revisions_quotation
    ON quotation_revisions(quotation_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_items_revision
    ON quotation_items(revision_id, line_number);
CREATE INDEX IF NOT EXISTS idx_audit_aggregate
    ON audit_events(aggregate_type, aggregate_id, sequence);
