"""User authentication: login, whoami, token semantics, and API-key coexistence.

The `client` fixture enables auth (api_keys=["test-key"]) and sets X-API-Key on
every request; tests that exercise token-only paths drop that header first.
"""
from api.auth import service, tokens


def _login(client, username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


def test_login_returns_token_and_me_works(client, runtime):
    service.create_user("alice", "s3cret")
    resp = _login(client, "alice", "s3cret")
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice" and body["token_type"] == "bearer" and body["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json() == {"username": "alice", "is_active": True}


def test_login_rejects_bad_password_and_unknown_user(client, runtime):
    service.create_user("alice", "s3cret")
    assert _login(client, "alice", "wrong").status_code == 401
    assert _login(client, "ghost", "whatever").status_code == 401


def test_token_alone_authorizes_protected_endpoint(client, runtime):
    service.create_user("alice", "s3cret")
    token = _login(client, "alice", "s3cret").json()["access_token"]
    client.headers.pop("X-API-Key", None)  # prove the token works without an API key
    assert client.get("/examinations/", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_api_key_still_authorizes_protected_endpoint(client, runtime):
    # coexistence: the default X-API-Key header (set by the fixture) is accepted
    assert client.get("/examinations/").status_code == 200


def test_missing_and_bad_credentials_rejected(client, runtime):
    client.headers.pop("X-API-Key", None)
    assert client.get("/examinations/").status_code == 401                       # nothing
    assert client.get("/examinations/", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_me_requires_a_token(client, runtime):
    client.headers.pop("X-API-Key", None)
    assert client.get("/auth/me").status_code == 401  # api key is not a "user"


def test_inactive_user_cannot_login(client, runtime):
    service.create_user("alice", "s3cret")
    service.set_active("alice", False)
    assert _login(client, "alice", "s3cret").status_code == 401


def test_password_change_invalidates_existing_token(client, runtime):
    service.create_user("alice", "s3cret")
    token = _login(client, "alice", "s3cret").json()["access_token"]
    service.set_password("alice", "different")  # bumps token_version
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 401


def test_expired_token_is_rejected(runtime):
    tok = tokens.create_token("alice", 0)
    assert tokens.read_token(tok, max_age_seconds=10_000) == {"u": "alice", "v": 0}
    assert tokens.read_token(tok, max_age_seconds=-1) is None  # treated as expired
