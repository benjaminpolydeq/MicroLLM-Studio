class TenantManager:
    def __init__(self):
        self.tenants = {}

    def register(self, tenant_id):
        self.tenants[tenant_id] = {"isolated": True}
