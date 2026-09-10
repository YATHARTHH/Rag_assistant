from database.sqlite import (
    check_embedding_cache,
    create_user,
    hash_password,
    save_embedding_cache,
    validate_password_strength,
    verify_user,
)


def test_password_strength_validation():
    assert validate_password_strength("Weak1") is False
    assert validate_password_strength("lowercaseonly123") is False
    assert validate_password_strength("NOCAPITALDIGITS") is False
    assert validate_password_strength("ValidPassword123") is True


def test_password_hashing():
    h1 = hash_password("MySecretPass1")
    h2 = hash_password("MySecretPass1")
    h3 = hash_password("DifferentPass1")
    assert h1 == h2
    assert h1 != h3


def test_user_creation_and_verification():
    username = "test_user_ci"
    password = "SecurePassword123"

    # Creation
    success = create_user(username, password, role="admin")
    assert success is True

    # Duplicate creation should fail
    duplicate = create_user(username, password)
    assert duplicate is False

    # Weak password creation should fail
    weak = create_user("weakuser", "weak")
    assert weak is False

    # Verification
    assert verify_user(username, password) is True
    assert verify_user(username, "WrongPassword123") is False
    assert verify_user("nonexistent_user", password) is False


def test_embedding_cache_operations():
    text = "Query string to cache vector embeddings."
    vector = [0.1, 0.2, 0.3, 0.4]

    assert check_embedding_cache(text) is None

    save_embedding_cache(text, vector)
    cached = check_embedding_cache(text)

    assert cached == vector
