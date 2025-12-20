def has_access(role, resource):
    return role in ["admin", "enterprise"]
