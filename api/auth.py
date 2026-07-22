import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, Header
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET", "SecureJWTSecretKeyForRAGTokenSigning12345=")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), x_api_key: Optional[str] = Header(None)) -> str:
    """
    Validates user credentials from either bearer token or programmatic API key header.
    """
    if x_api_key:
        if x_api_key == os.getenv("RAG_API_KEY", "rag_developer_key_123"):
            return "api_key_admin"
        raise HTTPException(status_code=401, detail="Invalid API Key.")
        
    if not token:
        raise HTTPException(status_code=401, detail="Authentication credentials required.")
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token has expired or is invalid.")
