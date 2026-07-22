import os
import base64
import hashlib
from cryptography.fernet import Fernet

FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    # Stable fallback key for zero-cost local environment
    FERNET_KEY = base64.urlsafe_b64encode(hashlib.sha256(b"SecurePayloadEncryptionKey123456789").digest())

def encrypt_text(text: str) -> str:
    """
    Encrypts clear text using AES-256 Fernet.
    """
    f = Fernet(FERNET_KEY)
    return f.encrypt(text.encode("utf-8")).decode("utf-8")

def decrypt_text(cipher_text: str) -> str:
    """
    Decrypts ciphertext using AES-256 Fernet.
    """
    f = Fernet(FERNET_KEY)
    return f.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
