REGISTRY: dict[str, type] = {}

def register(name: str):
    """Decorator that registers a policy class under ``name``."""
    def deco(cls):
        """Register ``cls`` inside the global registry and return it."""
        REGISTRY[name.upper()] = cls
        return cls
    return deco
