import jwt
import os

SECRET = os.getenv("JWT_SECRET")

def encode(payload):
    return jwt.encode(payload, SECRET, algorithm="HS256")
