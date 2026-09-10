import os

from fastapi.testclient import TestClient

# Ensure GROQ_API_KEY is set prior to importing api.main
os.environ["GROQ_API_KEY"] = "gsk_test_mock_key_for_api_tests"

from api.main import app

client = TestClient(app)


def test_signup_and_login_flow():
    # Test Signup
    signup_payload = {"username": "api_test_user", "password": "SecurePassword123"}
    signup_resp = client.post("/auth/signup", json=signup_payload)
    assert signup_resp.status_code in [200, 400]

    if signup_resp.status_code == 200:
        data = signup_resp.json()
        assert "successfully" in data.get("message", "").lower()

    # Test Login
    login_resp = client.post("/auth/login", json=signup_payload)
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "token" in login_data
    assert login_data["username"] == "api_test_user"


def test_login_invalid_credentials():
    login_payload = {"username": "nonexistent_api_user", "password": "WrongPassword123"}
    response = client.post("/auth/login", json=login_payload)
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]


def test_protected_route_unauthorized():
    # Chat endpoint without token or API key should return 401
    payload = {"query": "Hello", "username": "anonymous"}
    response = client.post("/chat", json=payload)
    assert response.status_code in [401, 403]
