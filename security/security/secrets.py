import os

def get_secret(name):
    return os.getenv(name)
