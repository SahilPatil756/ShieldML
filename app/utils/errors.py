"""
Custom exception classes and error responses for ShieldML.
"""

from __future__ import annotations

from fastapi import HTTPException, status


class ModelNotReadyException(HTTPException):
    def __init__(self, detail: str = "Model artifacts are not loaded or not ready"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class InvalidTransactionException(HTTPException):
    def __init__(self, detail: str = "Invalid transaction payload"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )
