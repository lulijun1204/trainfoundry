"""Domain errors raised by the local metadata store."""


class MetadataError(Exception):
    """Base class for metadata failures."""


class MetadataNotInitializedError(MetadataError):
    """Raised when an operation requires an initialized metadata database."""


class MetadataNotFoundError(MetadataError):
    """Raised when a requested metadata entity does not exist."""


class MetadataConflictError(MetadataError):
    """Raised when a uniqueness, foreign-key, or lifecycle rule is violated."""


class MetadataValidationError(MetadataError):
    """Raised when metadata input does not satisfy the public contract."""
