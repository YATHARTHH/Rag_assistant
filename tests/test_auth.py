import pytest
from fastapi import HTTPException
from jose import jwt

from api.auth import ALGORITHM, SECRET_KEY, create_access_token, get_current_user


def test_create_access_token():
    payload = {"username": "alice"}
    token = create_access_token(payload)
    assert isinstance(token, str)
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["username"] == "alice"
    assert "exp" in decoded

def test_get_current_user_with_valid_jwt():
    token = create_access_token({"username": "bob"})
    username = get_current_user(token=token)
    assert username == "bob"

def test_get_current_user_with_api_key():
    username = get_current_user(token=None, x_api_key="rag_developer_key_123")
    assert username == "api_key_admin"

def test_get_current_user_invalid_api_key():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=None, x_api_key="wrong_key")
    assert exc_info.value.status_code == 401
    assert "Invalid API Key" in exc_info.value.detail

def test_get_current_user_missing_credentials():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=None, x_api_key=None)
    assert exc_info.value.status_code == 401

def test_get_current_user_invalid_jwt():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token="invalid.jwt.token")
    assert exc_info.value.status_code == 401
