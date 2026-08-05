"""Signed, expiring session tokens (itsdangerous).

Stateless: the token carries the username and the user's ``token_version`` and is
authenticated by an HMAC signature over the secret key. There is no server-side
session store — a token is invalidated by expiry, by the user being deactivated,
or by bumping the user's ``token_version`` (e.g. on password change).
"""
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from api import runtime

_SALT = "morph-auth-token"


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(runtime.get_secret_key(), salt=_SALT)


def create_token(username: str, token_version: int) -> str:
    """Issue a signed token binding ``username`` to its current ``token_version``."""
    return _serializer().dumps({"u": username, "v": token_version})


def read_token(token: str, max_age_seconds: int) -> dict | None:
    """Return the token payload if the signature is valid and not older than
    ``max_age_seconds``; otherwise None. Caller still checks the user/version."""
    try:
        return _serializer().loads(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None
