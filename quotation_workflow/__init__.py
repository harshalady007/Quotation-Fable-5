"""V4 quotation workflow foundation.

The package deliberately exposes a transactional local domain service rather
than HTTP mutations. Workflow writes are not made available from the deployed
API; local tools use an explicitly configured SQLite file.
"""

from .domain import (
    WORKFLOW_SCHEMA_VERSION,
    WORKFLOW_VERSION,
    ApprovalDecision,
    ProjectStatus,
    QuotationStatus,
    UserRole,
    WorkflowAuthorizationError,
    WorkflowConflictError,
    WorkflowConfigurationError,
    WorkflowError,
    WorkflowIntegrityError,
    WorkflowNotFoundError,
    WorkflowStateError,
    WorkflowValidationError,
)
from .store import SQLiteWorkflowStore

__all__ = [
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_VERSION",
    "ApprovalDecision",
    "ProjectStatus",
    "QuotationStatus",
    "SQLiteWorkflowStore",
    "UserRole",
    "WorkflowAuthorizationError",
    "WorkflowConflictError",
    "WorkflowConfigurationError",
    "WorkflowError",
    "WorkflowIntegrityError",
    "WorkflowNotFoundError",
    "WorkflowStateError",
    "WorkflowValidationError",
]
