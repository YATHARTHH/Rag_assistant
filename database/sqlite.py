import os
import sqlite3
import hashlib
import json
from typing import List, Optional

USER_DB_PATH = "./users.db"

def init_user_db():
    """
    Initializes the local SQLite database and associated tables.
    """
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'readonly'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            feedback_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def validate_password_strength(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c.isupper() for c in password):
        return False
    return True

def hash_password(password: str) -> str:
    """
    Hashes a password using SHA-256.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def create_user(username: str, password: str, role: str = "readonly") -> bool:
    """
    Registers a new user. Returns True if successful, False if username exists or password is weak.
    """
    init_user_db()
    if not validate_password_strength(password):
        return False
    h = hash_password(password)
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, h, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    """
    Verifies user credentials.
    """
    init_user_db()
    h = hash_password(password)
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] == h:
        return True
    return False

def check_embedding_cache(text: str) -> Optional[List[float]]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT vector_json FROM embedding_cache WHERE text_hash = ?", (text_hash,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None

def save_embedding_cache(text: str, vector: List[float]):
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vector_json = json.dumps(vector)
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO embedding_cache (text_hash, vector_json) VALUES (?, ?)", (text_hash, vector_json))
        conn.commit()
        conn.close()
    except Exception:
        pass
