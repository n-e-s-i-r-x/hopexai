from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PublicAPIError(Exception):
    status_code: int
    message: str
    error_type: str
    code: str
    param: str | None = None

    def __str__(self) -> str:
        return self.message

    def as_payload(self) -> dict:
        error = {"message": self.message, "type": self.error_type, "code": self.code}
        if self.param is not None:
            error["param"] = self.param
        return {"error": error}


class ConfigurationUnavailable(RuntimeError):
    pass


class NoEligibleCandidate(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class UnsafeRequest(ValueError):
    pass


@dataclass(slots=True)
class UpstreamFailure(Exception):
    error_code: str
    status_code: int | None = None
    retry_safe: bool = False
    scope: str = "base"  # key | model | base | provider
    retry_after_seconds: int | None = None
    detail_for_logs: str | None = None

    def __str__(self) -> str:
        return self.error_code
