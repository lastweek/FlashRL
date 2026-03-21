def build_auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def build_timeout_header(timeout_seconds: int) -> dict[str, str]:
    return {"X-Timeout": str(timeout_seconds)}
